# Replaced with the current commit when building the wheels.
commit = "{{RAY_COMMIT_SHA}}"
version = "2.48.0"
package_time = "{{PACKAGE_TIME}}"

if __name__ == "__main__":
    print("%s %s %s" % (version, commit, package_time))
