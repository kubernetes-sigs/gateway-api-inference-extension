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

package podinfo

import (
	"context"
	"time"

	"sigs.k8s.io/gateway-api-inference-extension/api/v1alpha2"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend"
)

// NewScraperConfig returns a new ScraperConfig with the given args.
func NewScraperConfig(scrapeInterval time.Duration, scrapeTimeout time.Duration) *ScraperConfig {
	return &ScraperConfig{
		ScrapeInterval: scrapeInterval,
		ScrapeTimeout:  scrapeTimeout,
	}
}

// ScraperConfig defines the static configuration of a scraper.
type ScraperConfig struct {
	ScrapeInterval time.Duration
	ScrapeTimeout  time.Duration
}

// Scraper defines the required functions to scrape information from a pod.
type Scraper interface {
	// Name returns the name of the scraper.
	Name() string
	// Init returns a empty ScrapedData that will be stored upon initialization of the Scraper.
	// Each Scraper will have it's own data.
	InitData() ScrapedData
	// Scrape scrapes information from a pod.
	Scrape(ctx context.Context, pod *backend.Pod, port int) (ScrapedData, error)
	// ProcessResult process the returned object from Scrape function.
	// This function is used to update internal fields in the scraper struct before storing the ScrapedData in the PodInfo data map.
	ProcessResult(ctx context.Context, data ScrapedData)
}

// ScrapedData is a generic type for arbitrary scraperd data stored in PodInfo.
type ScrapedData interface {
	// Clone is an interface to make a copy of StateData.
	Clone() ScrapedData
}

type Datastore interface {
	PoolGet() (*v1alpha2.InferencePool, error)
}

// CloneScrapedData is a generic function that clones any struct that implements ScrapedData interface.
func CloneScrapedData[T ScrapedData](base T) T {
	copy := base.Clone()
	return copy.(T)
}
