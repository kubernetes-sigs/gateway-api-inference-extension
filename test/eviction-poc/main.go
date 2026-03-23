/*
Eviction PoC: Tests whether sending ImmediateResponse via ext_proc
causes Envoy to reset the upstream connection to the backend (vLLM).

Two test scenarios:
  1. /streaming  — backend actively streams tokens, ext_proc sends ImmediateResponse mid-stream
  2. /queued     — backend delays 10s before responding (simulates vLLM queue), ext_proc sends
                   ImmediateResponse after 2s while backend is still "queued"

Components:
  1. Mock backend (port 8080) — simulates vLLM with two endpoints
  2. ext_proc server (port 5001) — sends ImmediateResponse based on scenario
  3. Test client — sends request through Envoy (port 10000)

Usage:
  # Terminal 1: Start the PoC servers
  go run test/eviction-poc/main.go

  # Terminal 2: Start Envoy
  docker run --rm --network host --user $(id -u):$(id -g) \
    -v /path/to/test/eviction-poc/envoy.yaml:/etc/envoy/envoy.yaml:z \
    envoyproxy/envoy:v1.31-latest

  # Terminal 3: Test streaming eviction
  curl -v http://localhost:10000/streaming -d '{"model":"test"}'

  # Terminal 3: Test queued eviction
  curl -v http://localhost:10000/queued -d '{"model":"test"}'
*/
package main

import (
	"context"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"sync"
	"sync/atomic"
	"time"

	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	envoyTypePb "github.com/envoyproxy/go-control-plane/envoy/type/v3"
	"google.golang.org/grpc"
)

// --- Mock Backend (simulates vLLM) ---

type backendState struct {
	mu            sync.Mutex
	totalRequests int32
}

var backend = &backendState{}

// streamingHandler simulates vLLM actively generating tokens.
func streamingHandler(w http.ResponseWriter, r *http.Request) {
	reqNum := atomic.AddInt32(&backend.totalRequests, 1)
	log.Printf("[BACKEND] Request #%d received (STREAMING): %s %s", reqNum, r.Method, r.URL.Path)

	io.ReadAll(r.Body)
	r.Body.Close()

	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming not supported", http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.WriteHeader(http.StatusOK)
	flusher.Flush()

	for i := 0; i < 50; i++ {
		_, err := fmt.Fprintf(w, "data: {\"token\": \"word_%d\"}\n\n", i)
		if err != nil {
			log.Printf("[BACKEND] *** STREAMING DISCONNECT *** at token %d (request #%d): %v", i, reqNum, err)
			return
		}
		flusher.Flush()
		time.Sleep(200 * time.Millisecond)
	}

	log.Printf("[BACKEND] Request #%d completed normally (all 50 tokens)", reqNum)
}

// queuedHandler simulates a request stuck in vLLM's waiting queue.
// Delays 10s before sending any response — simulates prefill wait.
func queuedHandler(w http.ResponseWriter, r *http.Request) {
	reqNum := atomic.AddInt32(&backend.totalRequests, 1)
	log.Printf("[BACKEND] Request #%d received (QUEUED): %s %s — waiting 10s before responding...", reqNum, r.Method, r.URL.Path)

	io.ReadAll(r.Body)
	r.Body.Close()

	// Simulate being stuck in vLLM's queue — no response headers sent yet.
	// Use request context to detect if connection is closed during the wait.
	select {
	case <-time.After(10 * time.Second):
		log.Printf("[BACKEND] Request #%d queue wait finished, sending response", reqNum)
	case <-r.Context().Done():
		log.Printf("[BACKEND] *** QUEUED DISCONNECT *** request #%d: connection closed while queued: %v", reqNum, r.Context().Err())
		return
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	fmt.Fprintf(w, `{"response": "completed after queue wait"}`)
}

// --- ext_proc Server ---

type extProcServer struct {
	extProcPb.UnimplementedExternalProcessorServer
}

func (s *extProcServer) Process(srv extProcPb.ExternalProcessor_ProcessServer) error {
	log.Printf("[EXTPROC] New ext_proc stream started")

	var requestPath string
	responseBodyChunks := 0

	for {
		req, err := srv.Recv()
		if err == io.EOF {
			log.Printf("[EXTPROC] Stream EOF")
			return nil
		}
		if err != nil {
			log.Printf("[EXTPROC] Stream error: %v", err)
			return err
		}

		switch v := req.Request.(type) {
		case *extProcPb.ProcessingRequest_RequestHeaders:
			log.Printf("[EXTPROC] → RequestHeaders (end_of_stream=%v)", v.RequestHeaders.EndOfStream)

			// Extract the path to determine which scenario we're testing.
			for _, h := range v.RequestHeaders.Headers.Headers {
				if h.Key == ":path" {
					requestPath = string(h.RawValue)
				}
			}
			log.Printf("[EXTPROC]   path=%s", requestPath)

			err = srv.Send(&extProcPb.ProcessingResponse{
				Response: &extProcPb.ProcessingResponse_RequestHeaders{
					RequestHeaders: &extProcPb.HeadersResponse{},
				},
			})

		case *extProcPb.ProcessingRequest_RequestBody:
			log.Printf("[EXTPROC] → RequestBody (end_of_stream=%v)", v.RequestBody.EndOfStream)

			if requestPath == "/queued" {
				// QUEUED SCENARIO: Send the RequestBody response normally so Envoy
				// forwards to the backend. The backend will wait 10s (simulating queue).
				// Then send ImmediateResponse on the next ext_proc message we receive
				// (which will be ResponseHeaders once the backend eventually responds,
				// or we can try sending it proactively).
				log.Printf("[EXTPROC] Queued scenario: forwarding request, will evict on next ext_proc message")
				err = srv.Send(&extProcPb.ProcessingResponse{
					Response: &extProcPb.ProcessingResponse_RequestBody{
						RequestBody: &extProcPb.BodyResponse{},
					},
				})
				if err != nil {
					log.Printf("[EXTPROC] Error sending RequestBody response: %v", err)
					return err
				}

				// Now wait for the next message from Envoy. If the backend is truly
				// queued for 10s, we won't get ResponseHeaders for 10s.
				// This tests whether we CAN evict during this wait.
				// We'll use a goroutine to send ImmediateResponse after 2s,
				// racing against the Recv().
				log.Printf("[EXTPROC] Queued scenario: starting 2s eviction timer...")
				done := make(chan struct{})
				go func() {
					time.Sleep(2 * time.Second)
					log.Printf("[EXTPROC] *** SENDING IMMEDIATE_RESPONSE (503) *** for queued request")
					sendErr := srv.Send(&extProcPb.ProcessingResponse{
						Response: &extProcPb.ProcessingResponse_ImmediateResponse{
							ImmediateResponse: &extProcPb.ImmediateResponse{
								Status: &envoyTypePb.HttpStatus{
									Code: envoyTypePb.StatusCode_ServiceUnavailable,
								},
								Body: []byte("request evicted while queued"),
							},
						},
					})
					if sendErr != nil {
						log.Printf("[EXTPROC] Error sending ImmediateResponse: %v", sendErr)
					} else {
						log.Printf("[EXTPROC] ImmediateResponse sent successfully for queued request")
					}
					close(done)
				}()

				// Keep reading from the stream until it ends
				for {
					msg, recvErr := srv.Recv()
					if recvErr != nil {
						log.Printf("[EXTPROC] Stream ended: %v", recvErr)
						<-done
						return nil
					}
					log.Printf("[EXTPROC] Received message while waiting: %T", msg.Request)
				}
			}

			// STREAMING SCENARIO: Let the request through normally.
			err = srv.Send(&extProcPb.ProcessingResponse{
				Response: &extProcPb.ProcessingResponse_RequestBody{
					RequestBody: &extProcPb.BodyResponse{},
				},
			})

		case *extProcPb.ProcessingRequest_ResponseHeaders:
			log.Printf("[EXTPROC] → ResponseHeaders")
			err = srv.Send(&extProcPb.ProcessingResponse{
				Response: &extProcPb.ProcessingResponse_ResponseHeaders{
					ResponseHeaders: &extProcPb.HeadersResponse{},
				},
			})

		case *extProcPb.ProcessingRequest_ResponseBody:
			responseBodyChunks++
			log.Printf("[EXTPROC] → ResponseBody chunk #%d (end_of_stream=%v)",
				responseBodyChunks, v.ResponseBody.EndOfStream)

			if responseBodyChunks >= 3 {
				// STREAMING EVICTION: Send ImmediateResponse after 3 chunks
				log.Printf("[EXTPROC] *** SENDING IMMEDIATE_RESPONSE (503) *** after %d chunks", responseBodyChunks)
				err = srv.Send(&extProcPb.ProcessingResponse{
					Response: &extProcPb.ProcessingResponse_ImmediateResponse{
						ImmediateResponse: &extProcPb.ImmediateResponse{
							Status: &envoyTypePb.HttpStatus{
								Code: envoyTypePb.StatusCode_ServiceUnavailable,
							},
							Body: []byte("request evicted mid-stream"),
						},
					},
				})
				if err != nil {
					log.Printf("[EXTPROC] Error sending ImmediateResponse: %v", err)
				}
				log.Printf("[EXTPROC] ImmediateResponse sent, closing stream")
				return nil
			}

			// Normal pass-through
			err = srv.Send(&extProcPb.ProcessingResponse{
				Response: &extProcPb.ProcessingResponse_ResponseBody{
					ResponseBody: &extProcPb.BodyResponse{},
				},
			})
		}

		if err != nil {
			log.Printf("[EXTPROC] Error sending response: %v", err)
			return err
		}
	}
}

// suppress unused import
var _ = context.Background

func main() {
	// Start mock backend
	go func() {
		mux := http.NewServeMux()
		mux.HandleFunc("/streaming", streamingHandler)
		mux.HandleFunc("/queued", queuedHandler)
		log.Printf("[BACKEND] Starting on :8080")
		if err := http.ListenAndServe(":8080", mux); err != nil {
			log.Fatalf("[BACKEND] Failed: %v", err)
		}
	}()

	// Start ext_proc gRPC server
	lis, err := net.Listen("tcp", ":5001")
	if err != nil {
		log.Fatalf("[EXTPROC] Failed to listen: %v", err)
	}

	grpcServer := grpc.NewServer()
	extProcPb.RegisterExternalProcessorServer(grpcServer, &extProcServer{})

	log.Printf("[EXTPROC] Starting gRPC server on :5001")
	log.Printf("")
	log.Printf("=== Ready! Send requests through Envoy (port 10000) ===")
	log.Printf("")
	log.Printf("  Test 1 - Streaming eviction (backend actively generating):")
	log.Printf("    curl -v http://localhost:10000/streaming -d '{\"model\":\"test\"}'")
	log.Printf("")
	log.Printf("  Test 2 - Queued eviction (backend hasn't responded yet):")
	log.Printf("    curl -v http://localhost:10000/queued -d '{\"model\":\"test\"}'")
	log.Printf("")

	if err := grpcServer.Serve(lis); err != nil {
		log.Fatalf("[EXTPROC] Failed: %v", err)
	}
}
