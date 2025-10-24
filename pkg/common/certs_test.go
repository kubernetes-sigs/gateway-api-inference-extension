/*
Copyright 2025 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package common

import (
	"context"
	"crypto/rand"
	"crypto/rsa"
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"fmt"
	"math/big"
	"os"
	"path/filepath"
	"testing"
	"time"

	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
)

func TestCertReloader_InitialLoad(t *testing.T) {
	t.Parallel()

	// Create initial certificate
	certPEM1, keyPEM1, err := createTestCertificate(1)
	if err != nil {
		t.Fatalf("failed to create test certificate: %v", err)
	}

	// Setup K8s-style directory
	baseDir := setupK8sStyleCertDir(t, certPEM1, keyPEM1)

	// Load the initial certificate
	cert, err := tls.LoadX509KeyPair(filepath.Join(baseDir, "tls.crt"), filepath.Join(baseDir, "tls.key"))
	if err != nil {
		t.Fatalf("failed to load initial certificate: %v", err)
	}

	// Create cert reloader with debug logging
	ctx, cancel := context.WithCancel(logutil.NewTestLoggerIntoContext(context.Background()))
	defer cancel()

	reloader, err := NewCertReloader(ctx, baseDir, &cert)
	if err != nil {
		t.Fatalf("failed to create cert reloader: %v", err)
	}

	// Verify initial certificate is loaded
	loadedCert := reloader.Get()
	if loadedCert == nil {
		t.Fatal("expected certificate to be loaded")
	}

	// Verify it's the correct certificate by checking serial number
	if len(loadedCert.Certificate) == 0 {
		t.Fatal("loaded certificate has no certificate chain")
	}

	x509Cert, err := x509.ParseCertificate(loadedCert.Certificate[0])
	if err != nil {
		t.Fatalf("failed to parse loaded certificate: %v", err)
	}

	if x509Cert.SerialNumber.Int64() != 1 {
		t.Errorf("expected certificate serial number 1, got %d", x509Cert.SerialNumber.Int64())
	}
}

func TestCertReloader_MultipleUpdates(t *testing.T) {
	t.Parallel()

	initialSerialNumber := int64(1)

	// Create initial certificate
	certPEM1, keyPEM1, err := createTestCertificate(initialSerialNumber)
	if err != nil {
		t.Fatalf("failed to create test certificate: %v", err)
	}

	baseDir := setupK8sStyleCertDir(t, certPEM1, keyPEM1)

	// Load the initial certificate
	cert, err := tls.LoadX509KeyPair(filepath.Join(baseDir, "tls.crt"), filepath.Join(baseDir, "tls.key"))
	if err != nil {
		t.Fatalf("failed to load initial certificate: %v", err)
	}

	// Create cert reloader with debug logging
	ctx, cancel := context.WithCancel(logutil.NewTestLoggerIntoContext(context.Background()))
	defer cancel()

	reloader, err := NewCertReloader(ctx, baseDir, &cert)
	if err != nil {
		t.Fatalf("failed to create cert reloader: %v", err)
	}

	ticker := time.NewTicker(50 * time.Millisecond)
	defer ticker.Stop()

	// Perform multiple sequential updates
	for i := initialSerialNumber + 1; i <= 5; i++ {
		certPEM, keyPEM, err := createTestCertificate(i)
		if err != nil {
			t.Fatalf("failed to create test certificate %d: %v", i, err)
		}

		updateK8sStyleCerts(t, baseDir, certPEM, keyPEM)

		// Wait for reload
		timeout := time.After(10 * time.Second)

		reloaded := false
		for !reloaded {
			select {
			case <-timeout:
				t.Fatalf("timeout waiting for certificate reload to serial %d", i)
			case <-ticker.C:
				currentCert := reloader.Get()
				if len(currentCert.Certificate) > 0 {
					x509Cert, err := x509.ParseCertificate(currentCert.Certificate[0])
					if err != nil {
						t.Fatalf("failed to parse current certificate: %v", err)
					}
					if x509Cert.SerialNumber.Int64() == i {
						reloaded = true
					}
				}
			}
		}
	}

	// Verify final certificate
	finalCert := reloader.Get()
	x509FinalCert, err := x509.ParseCertificate(finalCert.Certificate[0])
	if err != nil {
		t.Fatalf("failed to parse final certificate: %v", err)
	}

	if x509FinalCert.SerialNumber.Int64() != 5 {
		t.Errorf("expected final certificate serial number 5, got %d", x509FinalCert.SerialNumber.Int64())
	}
}

func TestCertReloader_ErrorHandling(t *testing.T) {
	t.Parallel()

	// Create initial valid certificate
	certPEM1, keyPEM1, err := createTestCertificate(1)
	if err != nil {
		t.Fatalf("failed to create test certificate: %v", err)
	}

	// Setup K8s-style directory
	baseDir := setupK8sStyleCertDir(t, certPEM1, keyPEM1)

	// Load the initial certificate
	cert, err := tls.LoadX509KeyPair(filepath.Join(baseDir, "tls.crt"), filepath.Join(baseDir, "tls.key"))
	if err != nil {
		t.Fatalf("failed to load initial certificate: %v", err)
	}

	// Create cert reloader with debug logging
	ctx, cancel := context.WithCancel(logutil.NewTestLoggerIntoContext(context.Background()))
	defer cancel()

	reloader, err := NewCertReloader(ctx, baseDir, &cert)
	if err != nil {
		t.Fatalf("failed to create cert reloader: %v", err)
	}

	// Verify initial certificate
	initialCert := reloader.Get()
	x509InitialCert, err := x509.ParseCertificate(initialCert.Certificate[0])
	if err != nil {
		t.Fatalf("failed to parse initial certificate: %v", err)
	}

	// Simulate an invalid certificate update (mismatched cert and key)
	certPEM2, _, err := createTestCertificate(2)
	if err != nil {
		t.Fatalf("failed to create test certificate 2: %v", err)
	}
	_, keyPEM3, err := createTestCertificate(3)
	if err != nil {
		t.Fatalf("failed to create test certificate 3: %v", err)
	}

	// Update with mismatched cert and key (should fail to load)
	updateK8sStyleCerts(t, baseDir, certPEM2, keyPEM3)

	// Wait a bit to allow the reload attempt
	time.Sleep(2 * time.Second)

	// Verify that the old certificate is still loaded (reload should have failed)
	currentCert := reloader.Get()
	x509CurrentCert, err := x509.ParseCertificate(currentCert.Certificate[0])
	if err != nil {
		t.Fatalf("failed to parse current certificate: %v", err)
	}

	if x509CurrentCert.SerialNumber.Int64() != x509InitialCert.SerialNumber.Int64() {
		t.Errorf("expected certificate to remain unchanged on reload error, but serial number changed from %d to %d",
			x509InitialCert.SerialNumber.Int64(), x509CurrentCert.SerialNumber.Int64())
	}
}

// createTestCertificate generates a test TLS certificate and key pair with a given serial number.
// The serial number is used to uniquely identify different certificates in tests.
func createTestCertificate(serialNum int64) (certPEM, keyPEM []byte, err error) {
	priv, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to generate private key: %w", err)
	}

	notBefore := time.Now()
	notAfter := notBefore.Add(24 * time.Hour)

	template := x509.Certificate{
		SerialNumber: big.NewInt(serialNum),
		Subject: pkix.Name{
			Organization: []string{"Test Org"},
			CommonName:   "test-cert",
		},
		NotBefore:             notBefore,
		NotAfter:              notAfter,
		KeyUsage:              x509.KeyUsageKeyEncipherment | x509.KeyUsageDigitalSignature,
		ExtKeyUsage:           []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
		BasicConstraintsValid: true,
	}

	certDER, err := x509.CreateCertificate(rand.Reader, &template, &template, &priv.PublicKey, priv)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to create certificate: %w", err)
	}

	certPEM = pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: certDER})

	privBytes, err := x509.MarshalPKCS8PrivateKey(priv)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to marshal private key: %w", err)
	}
	keyPEM = pem.EncodeToMemory(&pem.Block{Type: "PRIVATE KEY", Bytes: privBytes})

	return certPEM, keyPEM, nil
}

// setupK8sStyleCertDir creates a directory structure that mimics Kubernetes secret volume mounts.
// It creates:
// - A timestamped data directory (..2025_01_15_12_00_00.123456789/)
// - A ..data symlink pointing to the timestamped directory
// - tls.crt and tls.key symlinks pointing to ..data/tls.crt and ..data/tls.key
func setupK8sStyleCertDir(t *testing.T, certPEM, keyPEM []byte) string {
	t.Helper()

	baseDir := t.TempDir()

	// Create initial timestamped directory
	timestamp := time.Now().Format("..2006_01_02_15_04_05.000000000")
	dataDir := filepath.Join(baseDir, timestamp)
	if err := os.MkdirAll(dataDir, 0755); err != nil {
		t.Fatalf("failed to create data directory: %v", err)
	}

	// Write certificates to the timestamped directory
	certPath := filepath.Join(dataDir, "tls.crt")
	if err := os.WriteFile(certPath, certPEM, 0644); err != nil {
		t.Fatalf("failed to write certificate: %v", err)
	}

	keyPath := filepath.Join(dataDir, "tls.key")
	if err := os.WriteFile(keyPath, keyPEM, 0600); err != nil {
		t.Fatalf("failed to write key: %v", err)
	}

	// Create ..data symlink pointing to the timestamped directory
	dotDataLink := filepath.Join(baseDir, "..data")
	if err := os.Symlink(timestamp, dotDataLink); err != nil {
		t.Fatalf("failed to create ..data symlink: %v", err)
	}

	// Create tls.crt and tls.key symlinks pointing through ..data
	tlsCrtLink := filepath.Join(baseDir, "tls.crt")
	if err := os.Symlink(filepath.Join("..data", "tls.crt"), tlsCrtLink); err != nil {
		t.Fatalf("failed to create tls.crt symlink: %v", err)
	}

	tlsKeyLink := filepath.Join(baseDir, "tls.key")
	if err := os.Symlink(filepath.Join("..data", "tls.key"), tlsKeyLink); err != nil {
		t.Fatalf("failed to create tls.key symlink: %v", err)
	}

	return baseDir
}

// updateK8sStyleCerts simulates a Kubernetes secret update by:
// 1. Creating a new timestamped directory with new certificates
// 2. Atomically updating the ..data symlink to point to the new directory
// This mimics how Kubernetes updates secret volumes.
func updateK8sStyleCerts(t *testing.T, baseDir string, newCertPEM, newKeyPEM []byte) {
	t.Helper()

	// Create new timestamped directory
	newTimestamp := time.Now().Format("..2006_01_02_15_04_05.000000000")
	newDataDir := filepath.Join(baseDir, newTimestamp)
	if err := os.MkdirAll(newDataDir, 0755); err != nil {
		t.Fatalf("failed to create new data directory: %v", err)
	}

	// Write new certificates
	certPath := filepath.Join(newDataDir, "tls.crt")
	if err := os.WriteFile(certPath, newCertPEM, 0644); err != nil {
		t.Fatalf("failed to write new certificate: %v", err)
	}

	keyPath := filepath.Join(newDataDir, "tls.key")
	if err := os.WriteFile(keyPath, newKeyPEM, 0600); err != nil {
		t.Fatalf("failed to write new key: %v", err)
	}

	// Atomically update ..data symlink
	// In Kubernetes, this is done atomically by creating a new symlink and renaming it
	dotDataLink := filepath.Join(baseDir, "..data")
	dotDataTmp := filepath.Join(baseDir, "..data_tmp")

	// Create temporary symlink
	if err := os.Symlink(newTimestamp, dotDataTmp); err != nil {
		t.Fatalf("failed to create temporary ..data symlink: %v", err)
	}

	// Atomically replace the old symlink
	if err := os.Rename(dotDataTmp, dotDataLink); err != nil {
		t.Fatalf("failed to update ..data symlink: %v", err)
	}
}
