#!/bin/sh -xeu

if [ -z "${VERSION:-}" ]; then
    ARCHIVE="oschecktool"
else 
    ARCHIVE="oschecktool-${VERSION}"
fi


SCRIPT_DIR=$(cd $(dirname "$0") ; pwd)
ROOT_DIR=$(cd "${SCRIPT_DIR}/.." ; pwd)


OUT_DIR="${ROOT_DIR}/output/${ARCHIVE}"
PACKAGE="${OUT_DIR}.tar.gz"



build() {
    CGO_ENABLED=0 go build -C "${ROOT_DIR}" -trimpath -buildmode=pie -ldflags="-w -s" -o "${OUT_DIR}/oschecktool"
    chmod 550 "${OUT_DIR}/oschecktool"
}

package() {
    cp -r "${ROOT_DIR}/plugin" "${OUT_DIR}/"
    cp -r "${ROOT_DIR}/conf" "${OUT_DIR}/"
    tar -czvf "${PACKAGE}" -C "${OUT_DIR}" .
}

prepare() {
    rm -rf "${OUT_DIR}"
    mkdir -p "${OUT_DIR}"
}
main() {
    prepare
    build
    package
}

main "$@"