#./colocation-manager --v=4 --refresh-nodes-interval=10s --refresh-pods-interval=5s 2>&1  | tee output.log | grep "toolt-validation"
#
#kubectl proxy

NODENAME=192.168.0.120

#curl --header "Content-Type: application/json-patch+json"   --request PATCH   --data '[{"op": "add", "path": "/status/capacity/example.com~1dongle", "value": "4"}]'   http://localhost:8001/api/v1/nodes/192.168.0.120/status
curl --header "Content-Type: application/json-patch+json" --request PATCH --data '[{"op": "remove", "path": "/status/capacity/example.com~1dongle"}]' http://localhost:8001/api/v1/nodes/${NODENAME}/status

curl --header "Content-Type: application/json-patch+json" --request PATCH --data '[{"op": "remove", "path": "/status/capacity/kubernetes.io~1be-cpu"}]' http://localhost:8001/api/v1/nodes/${NODENAME}/status
curl --header "Content-Type: application/json-patch+json" --request PATCH --data '[{"op": "remove", "path": "/status/allocatable/kubernetes.io~1be-cpu"}]' http://localhost:8001/api/v1/nodes/${NODENAME}/status

curl --header "Content-Type: application/json-patch+json" --request PATCH --data '[{"op": "remove", "path": "/status/capacity/kubernetes.io~1be-memory"}]' http://localhost:8001/api/v1/nodes/${NODENAME}/status
curl --header "Content-Type: application/json-patch+json" --request PATCH --data '[{"op": "remove", "path": "/status/allocatable/kubernetes.io~1be-memory"}]' http://localhost:8001/api/v1/nodes/${NODENAME}/status


./colocation-manager --v=2 --refresh-nodes-interval=15s --refresh-pods-interval=15s --be-recommend-min-cpu-millicores=100 --be-recommend-min-memory-mb=500  --save-checkpoint-period=10s  --checkpoint-gc-interval=60s --aggregate-state-gc-interval=60s 2>&1  | tee output.log
kubectl get containercheckpoints -n openfuyao-colocation| grep -v NAME | awk '{print }' | xargs -i kubectl delete containercheckpoint -n openfuyao-colocation {}
./colocation-manager --v=2 --refresh-nodes-interval=15s --refresh-pods-interval=15s --be-recommend-min-cpu-millicores=100 --be-recommend-min-memory-mb=500  --save-checkpoint-period=10s  --checkpoint-gc-interval=60s --aggregate-state-gc-interval=60s 2>&1  | tee output.log
