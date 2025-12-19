package datalayer

type DataStore interface {
	PodList(func(Endpoint) bool) []Endpoint
	PoolGet() (*EndpointPool, error)
}
