# Certificate Generation and Mounting for openFuyao Core Components

This document aims to provide a general solution for certificate configuration of core components on the openFuyao container platform after installation using the openFuyao installation and deployment tool. This solution requires users to configure all certificates in the secret of the `openfuyao-system` namespace for use by each component.

# Generate Certificates for Core Components

## File Description

### getCA.bash

Get `CA` from the `openfuyao-system` namespace (including the original and base64 decoded).

```bash
#!/bin/bash
# Get ca.crt
kubectl get secret -n openfuyao-system openfuyao-system-root-ca -oyaml | grep "ca.crt: " | cut -d' ' -f4 > original_ca.crt
# base64 decode
cat original_ca.crt | base64 -d > base64_ca.crt

# Get ca.key
kubectl get secret -n openfuyao-system openfuyao-system-root-ca -oyaml | grep "ca.key: " | cut -d' ' -f4 > original_ca.key
# base64 decode
cat original_ca.key | base64 -d > base64_ca.key

cp base64_ca.crt ca.crt
cp base64_ca.key ca.key
```

### createCert.bash

Generate certificates for services using the specified `CA`.

```bash
#!/bin/bash
# 1. Set certificate generation target
set -e

NAMESPACE="openfuyao-system"
SERVICE_NAME="monitoring-service"
# SERVICE_NAME="user-management-operator"
# SERVICE_NAME="web-terminal-service"
# SERVICE_NAME="application-management-service"
# SERVICE_NAME="plugin-management-service"
# SERVICE_NAME="marketplace-service"
SECRET_NAME="${SERVICE_NAME}-tls"

# 2. Regenerate webhook certificate for specified target
echo "Start regenerating webhook certificate..."

# 3. Generate server private key
echo "Generate server private key..."
openssl genrsa -out ${SERVICE_NAME}.key 4096

# 4. Create server certificate configuration file
echo "Create certificate configuration file..."
cat > ${SERVICE_NAME}.conf <<EOF
[req]
distinguished_name = req_distinguished_name
req_extensions = v3_req
prompt = no

[req_distinguished_name]
CN = ${SERVICE_NAME}.${NAMESPACE}.svc

[v3_req]
keyUsage = keyEncipherment, dataEncipherment
extendedKeyUsage = serverAuth, clientAuth
subjectAltName = @alt_names

[alt_names]
DNS.1 = ${SERVICE_NAME}
DNS.2 = ${SERVICE_NAME}.${NAMESPACE}
DNS.3 = ${SERVICE_NAME}.${NAMESPACE}.svc
DNS.4 = ${SERVICE_NAME}.${NAMESPACE}.svc.cluster.local
EOF

# 5. Generate certificate signing request
echo "Generate certificate signing request..."
openssl req -new -key ${SERVICE_NAME}.key -out ${SERVICE_NAME}.csr -config ${SERVICE_NAME}.conf

# 6. Generate server certificate
echo "Generate server certificate..."
openssl x509 -req -in ${SERVICE_NAME}.csr -CA ca.crt -CAkey ca.key -CAcreateserial -out ${SERVICE_NAME}.crt -days 1095 -sha256 -extensions v3_req -extfile ${SERVICE_NAME}.conf

# 7. Verify certificate
echo "Verify generated certificate..."
echo "Certificate subject:"
openssl x509 -in ${SERVICE_NAME}.crt -text -noout | grep "Subject:"
echo "Certificate SAN:"
openssl x509 -in ${SERVICE_NAME}.crt -text -noout | grep -A 10 "Subject Alternative Name"

# 8. Check if namespace exists
echo "Check namespace ${NAMESPACE}..."
if ! kubectl get namespace ${NAMESPACE} >/dev/null 2>&1; then
    echo "Create namespace ${NAMESPACE}..."
    kubectl create namespace ${NAMESPACE}
fi

# 9. Delete old Secret (if exists)
echo "Delete old certificate Secret..."
kubectl delete secret ${SECRET_NAME} -n ${NAMESPACE} --ignore-not-found=true

# 10. Create new Secret, using correct key names matching Helm template
echo "Create new certificate Secret..."
kubectl create secret generic ${SECRET_NAME} -n ${NAMESPACE} \
    --from-file=ca.crt=ca.crt \
    --from-file=tls.key=${SERVICE_NAME}.key \
    --from-file=tls.crt=${SERVICE_NAME}.crt

# 11. Verify Secret creation
echo "Verify Secret creation..."
kubectl get secret ${SECRET_NAME} -n ${NAMESPACE} -o yaml

# 12. Encode certificate to base64 (for webhook configuration)
echo "Encode certificate..."
# CA_BUNDLE=$(cat ca.crt | base64 | tr -d '\n')
CA_BUNDLE=$(cat ca.crt | tr -d '\n')

echo ""
echo "==============================================="
echo "Certificate regeneration complete!"
echo "==============================================="
echo ""
echo "Secret information:"
echo "- Namespace: ${NAMESPACE}"
echo "- Secret name: ${SECRET_NAME}"
echo "- Included keys:"
echo "  * ca.crt (CA certificate)"
echo "  * tls.key (server private key)"
echo "  * tls.crt (server certificate)"
echo ""
echo "Mapping in Helm template:"
echo "- ca.crt -> /ssl/ca.pem"
echo "- tls.key -> /ssl/server.key"  
echo "- tls.crt -> /ssl/server.crt"
echo ""
echo "New CA Bundle (for webhook configuration):"
echo "${CA_BUNDLE}"
echo ""
echo "You can now verify certificate mounting using the following command:"
echo "kubectl exec -it <pod-name> -n ${NAMESPACE} -- ls -la /ssl/"
echo ""
echo "Restart related Pod to apply new certificate:"
echo "kubectl rollout restart deployment/${SERVICE_NAME} -n ${NAMESPACE}"
```

## Usage Instructions

1. Place `getCA.bash` and `createCert.bash` in the user-specified folder (subsequent operations must be performed in this folder).
2. Set the above files as executable `chmod u+x getCA.bash`.
3. Run `getCA.bash` to get CA.
4. Run `createCert.bash` to generate certificates (modify `SERVICE_NAME` to generate certificates for different services).

    Generate certificates for the following 6 components:

    ```bash
    SERVICE_NAME="monitoring-service"
    # SERVICE_NAME="user-management-operator"
    # SERVICE_NAME="web-terminal-service"
    # SERVICE_NAME="application-management-service"
    # SERVICE_NAME="plugin-management-service"
    # SERVICE_NAME="marketplace-service"
    ```

# Certificate Mounting

## Task 1: Modify the `console-service-config` ConfigMap in the `openfuyao-system` Namespace

> **NOTE:** The first YAML snippet below shows the configuration before modification (services are accessed over the HTTP protocol), and the second YAML snippet shows the target configuration after modification. The HTTP protocol has security risks. HTTPS is recommended.

### Operation Steps

1. Execute `kubectl edit cm -n openfuyao-system console-service-config` to modify the following parts of the ConfigMap:
   
   ```yaml
   apiVersion: v1
   data:
     alert-host: http://alertmanager-main.monitoring.svc.cluster.local:9093
     application-management-host: http://application-management-service.openfuyao-system.svc.cluster.local
     console-service-host: https://console-service.openfuyao-system.svc.cluster.local:443
     console-website-host: https://console-website.openfuyao-system.svc.cluster.local:80
     insecure-skip-verify: "false"
     marketplace-host: http://marketplace-service.openfuyao-system.svc.cluster.local
     monitoring-host: http://monitoring-service.openfuyao-system.svc.cluster.local:80
     oauth-server-host: https://oauth-server.openfuyao-system.svc.cluster.local:9096
     plugin-management-host: http://plugin-management-service.openfuyao-system.svc.cluster.local
     server-name: ""
     user-management-host: http://user-management-operator.openfuyao-system.svc.cluster.local:80
     webterminal-host: http://web-terminal-service.openfuyao-system.svc.cluster.local:9072
   kind: ConfigMap
   ...
   ```

2. The modified result should be:
   
   ```yaml
   apiVersion: v1
   data:
     alert-host: http://alertmanager-main.monitoring.svc.cluster.local:9093
     application-management-host: https://application-management-service.openfuyao-system.svc.cluster.local:80
     console-service-host: https://console-service.openfuyao-system.svc.cluster.local:80
     console-website-host: https://console-website.openfuyao-system.svc.cluster.local:80
     insecure-skip-verify: "false"
     marketplace-host: https://marketplace-service.openfuyao-system.svc.cluster.local:80
     monitoring-host: https://monitoring-service.openfuyao-system.svc.cluster.local:80
     oauth-server-host: https://oauth-server.openfuyao-system.svc.cluster.local:9096
     plugin-management-host: https://plugin-management-service.openfuyao-system.svc.cluster.local:80
     server-name: ""
     user-management-host: https://user-management-operator.openfuyao-system.svc.cluster.local:80
     webterminal-host: https://web-terminal-service.openfuyao-system.svc.cluster.local:9072
   kind: ConfigMap
   ...
   ```

3. Restart the `console-service` Pod, execute `kubectl delete po -n openfuyao-system console-service-xxxxxxx-xxxxx`, and observe that the corresponding Pod restarts and is running.

## Task 2: Mount Certificates for `user-management-operator` and `web-terminal-service` to Their Own Containers

### Operation Steps

Here is an example of mounting `user-management-operator-tls` inside the container:

1. Execute the command `kubectl edit deployment/user-management-operator -n openfuyao-system` to modify the deployment `yaml`.
2. Find `spec.template.spec.containers` for the `user-management-operator` container configuration and add `volumeMounts` configuration.

    ```yaml
                volumeMounts:
                - name: user-management-operator-tls
                  mountPath: /ssl/ca.pem
                  subPath: ca.crt
                - name: user-management-operator-tls
                  readOnly: true
                  mountPath: /ssl/server.key
                  subPath: tls.key
                - name: user-management-operator-tls
                  readOnly: true
                  mountPath: /ssl/server.crt
                  subPath: tls.crt
    ```

3. Add the following configuration to `volumes`.

    ```yaml
                - name: user-management-operator-tls
                  secret:
                    defaultMode: 0600
                    secretName: user-management-operator-tls
    ```

4. Save and exit, then execute the command `kubectl describe deployment/user-management-operator -n openfuyao-system` to check if the mounting was successful.

## Task 3: Mount Certificates for `monitoring-service`, `application-management-service`, `plugin-management-service`, and `marketplace-service` to the `oauth-proxy` Container

### Operation Steps

Here is an example of mounting `monitoring-service-tls` inside the container:

1. Execute the command `kubectl edit deployment/monitoring-service -n openfuyao-system` to modify the deployment `yaml`. After executing the command, you will enter file modification.
2. Find `spec.template.spec.containers` for the `oauth-proxy` container's `args` and add the content noted in the code block below.

    ```yaml
        apiVersion: apps/v1
        kind: Deployment
        metadata:
          annotations:
            deployment.kubernetes.io/revision: "1"
            meta.helm.sh/release-name: monitoring-service
            meta.helm.sh/release-namespace: openfuyao-system
          creationTimestamp: "2025-09-10T02:19:56Z"
          generation: 1
          labels:
            app.kubernetes.io/managed-by: Helm
          name: monitoring-service
          namespace: openfuyao-system
          resourceVersion: "3115"
          uid: 76f1e52d-5153-485b-9607-2c987ce2f0f2
        spec:
          progressDeadlineSeconds: 600
          replicas: 1
          revisionHistoryLimit: 10
          selector:
            matchLabels:
              app: monitoring-service
          strategy:
            rollingUpdate:
              maxSurge: 25%
              maxUnavailable: 25%
            type: RollingUpdate
          template:
            metadata:
              creationTimestamp: null
              labels:
                app: monitoring-service
            spec:
              containers:
              - args:
                - --https-address=:9093 # Need to modify
                - --http-address= # Need to modify
                - --email-domain=*
                - --provider=openfuyao
                - --client-id=oauth-proxy
                - --client-secret=SECRETTS
                - --tls-cert=/ssl/server.crt # Need to modify
                - --tls-key=/ssl/server.key # Need to modify
                - --tls-client-ca=/ssl/ca.pem # Need to modify
                - --upstream=http://localhost:9083  # pod-internal trusted domain network access, http is acceptable
                - '--openfuyao-delegate-urls={"/":{"resource": "services/proxy", "group":
                  ""}}'
                - --redirect-url=/
                - --login-url=/oauth2/oauth/authorize
                - --redeem-url=/oauth2/oauth/token
                - --root-prefix=/
                - --cookie-httponly
                image: harbor.openfuyao.com/openfuyao/oauth-proxy:latest
        ...
    ```

3. Add `volumeMounts` configuration for the specified container (oauth-proxy).

    ```yaml
                volumeMounts:
                - name: monitoring-service-tls
                  mountPath: /ssl/ca.pem
                  subPath: ca.crt
                - name: monitoring-service-tls
                  readOnly: true
                  mountPath: /ssl/server.key
                  subPath: tls.key
                - name: monitoring-service-tls
                  readOnly: true
                  mountPath: /ssl/server.crt
                  subPath: tls.crt
    ```

4. Add the following configuration to the `oauth-proxy` container's `volumes`.

    ```yaml
                - name: monitoring-service-tls
                  secret:
                    defaultMode: 0600
                    secretName: monitoring-service-tls
    ```

5. Save and exit, then execute the command `kubectl describe deployment/monitoring-service -n openfuyao-system` to check if the mounting was successful.

## Task 4: Mount Certificates for `application-management-service` and `marketplace-service` to Their Own Containers for TLS Protection of RPC Communication

### Operation Steps

Here is an example of mounting `application-management-service-tls` inside the container:

1. Execute the command `kubectl edit deploy application-management-service -n openfuyao-system` to modify the deployment `yaml`. After executing the command, you will enter file modification.
2. Add `volumeMounts` configuration for the specified container (application-management-service) (using the same certificate mounted in Task 3).

    ```yaml
                volumeMounts:
                - name: application-management-service-tls
                  mountPath: /ssl/ca.pem
                  subPath: ca.crt
                - name: application-management-service-tls
                  readOnly: true
                  mountPath: /ssl/server.key
                  subPath: tls.key
                - name: application-management-service-tls
                  readOnly: true
                  mountPath: /ssl/server.crt
                  subPath: tls.crt
    ```

3. Save and exit, then execute the command `kubectl describe deploy application-management-service -n openfuyao-system` to check if the mounting was successful.
