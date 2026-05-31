# RAG Service Deployment

## Prerequisites

- A Kubernetes cluster with `kubectl` configured
- Qdrant and Ollama deployed and reachable from the `rag-service` namespace
- Docker (or Podman) running locally if building images by hand

## 1. Pull the published image

The image is built, signed, and published by the GitHub Actions workflow (`.github/workflows/build-test.yml`) to GitHub Container Registry on every push to `main`.

```bash
docker pull ghcr.io/cristianciobanu/rag-service:latest
```

The image signature can be verified with cosign:

```bash
cosign verify ghcr.io/cristianciobanu/rag-service:latest \
  --certificate-identity-regexp "https://github.com/cristianciobanu/zero-trust-architecture-for-ai-apps/.*" \
  --certificate-oidc-issuer https://token.actions.githubusercontent.com
```

## 2. Build locally (optional)

```bash
cd rag-service
docker build --platform linux/amd64 -t rag-service:local .
```

The Dockerfile bakes the `all-MiniLM-L6-v2` embedding model into the image at build time, so the running container does not need internet access at startup.

## 3. Apply the manifests

```bash
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/serviceaccount.yaml
kubectl apply -f k8s/networkpolicy.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

kubectl wait --namespace rag-service --for=condition=ready pod \
  -l app.kubernetes.io/name=rag-service --timeout=120s
```

For TLS exposure through nginx-ingress with cert-manager:

```bash
kubectl apply -f k8s/certificate.yaml
kubectl apply -f k8s/ingress.yaml
```

The ingress assumes `cert-manager` is installed in the cluster with a `letsencrypt-prod` ClusterIssuer.

## 4. Verify

```bash
kubectl port-forward -n rag-service svc/rag-service 8000:8000

# Health check
curl http://localhost:8000/health

# Query (no documents ingested yet — see ingest.py)
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the net revenue?"}'
```

## Notes

- The service is stateless, so no PersistentVolumeClaim is needed and a `RollingUpdate` strategy is safe.
- JWT validation against Keycloak requires `KEYCLOAK_URL`, `KEYCLOAK_REALM`, and `KEYCLOAK_CLIENT_ID` to point at a reachable identity provider (the deployment manifest includes placeholder values).
- The NetworkPolicies enforce default-deny ingress and egress, with explicit allow rules for Qdrant, Ollama, the identity provider, and DNS.
