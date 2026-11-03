# Logging Extension Frontend

## Project Run

```bash
# Install dependencies
npm install

# Run localhost:7777
npm run start

# Build
npm run build
```

## Image Build

```bash
docker build -t cr.openfuyao.cn/openfuyao/logging-website:latest -f ./build/Dockerfile .
```

## Helm Deployment

```bash
helm install {release} ./chart -n {namespace}
```
