kubectl get containercheckpoints -n openfuyao-colocation | grep -v NAME | awk '{print $1}' | xargs -i kubectl delete containercheckpoint -n openfuyao-colocation {}
