# Robot-side image: libfranka 0.9.2 + the C++ data collector.
#
# libfranka 0.9.2 is the LAST release that talks to a classic Panda (FER); 0.10 and
# later are FR3-only and will refuse the connection. Ubuntu 20.04 is the officially
# supported pairing, and 0.9.2 needs only Poco + Eigen -- no fmt, no Pinocchio.
#
# The PREEMPT_RT kernel lives on the HOST. Containers share the host kernel, so there
# is no such thing as a real-time image; these flags only grant permission to request
# real-time scheduling:
#
#   docker run --rm -it --network=host --cap-add=SYS_NICE \
#     --ulimit rtprio=99 --ulimit rttime=-1 --ulimit memlock=-1 \
#     -v $PWD/data:/data fpi-robot:0.9.2 fpi_check --ip 172.16.0.2
#
# --network=host is effectively mandatory: FCI is TCP 1337 plus UDP 1338 at 1 kHz, and
# bridge NAT reliably produces communication_constraints_violation.
#
# Do NOT mount /tmp noexec: Robot::loadModel() downloads a shared object from the
# Control box at runtime, writes it into /tmp and dlopen()s it.

# ---------------------------------------------------------------- libfranka
FROM ubuntu:20.04 AS libfranka-build
ARG DEBIAN_FRONTEND=noninteractive
ARG LIBFRANKA_VERSION=0.9.2

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git ca-certificates \
        libpoco-dev libeigen3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt
# --recursive is mandatory: 0.9.2 vendors its `common` submodule, which carries the
# robot/gripper server version constants. Without it CMake fails even with tests off.
RUN git clone --recursive --depth 1 --branch ${LIBFRANKA_VERSION} \
        https://github.com/frankaemika/libfranka.git

WORKDIR /opt/libfranka/build
RUN cmake -DCMAKE_BUILD_TYPE=Release \
          -DBUILD_TESTS=OFF -DBUILD_EXAMPLES=OFF \
          -DCMAKE_INSTALL_PREFIX=/usr/local .. \
 && cmake --build . -- -j"$(nproc)" \
 && cpack -G DEB \
 && dpkg -i libfranka*.deb

# ---------------------------------------------------------------- collector
FROM libfranka-build AS app-build
COPY cpp/          /src/cpp/
COPY CMakeLists.txt /src/CMakeLists.txt
WORKDIR /src/build
RUN cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_ROBOT_COLLECTOR=ON /src \
 && cmake --build . -- -j"$(nproc)"

# ---------------------------------------------------------------- runtime
FROM ubuntu:20.04 AS runtime
ARG DEBIAN_FRONTEND=noninteractive
ARG USER_UID=1000
ARG USER_GID=1000

# Install the libfranka .deb through apt so its declared dependencies (the Poco
# runtime libraries) are resolved automatically. Hand-listing them is brittle: the
# Poco package names carry an ABI version suffix that changes between Ubuntu releases.
COPY --from=libfranka-build /opt/libfranka/build/libfranka*.deb /tmp/
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates /tmp/libfranka*.deb \
 && rm -f /tmp/*.deb \
 && rm -rf /var/lib/apt/lists/* \
 && ldconfig

COPY --from=app-build /src/build/cpp/fpi_check          /usr/local/bin/
COPY --from=app-build /src/build/cpp/fpi_move_to        /usr/local/bin/
COPY --from=app-build /src/build/cpp/fpi_static_poses   /usr/local/bin/
COPY --from=app-build /src/build/cpp/fpi_run_trajectory /usr/local/bin/

RUN groupadd -g ${USER_GID} fpi 2>/dev/null || true \
 && useradd -m -u ${USER_UID} -g ${USER_GID} fpi 2>/dev/null || true \
 && mkdir -p /data && chown -R ${USER_UID}:${USER_GID} /data
USER ${USER_UID}:${USER_GID}

ENV FPI_DATA_DIR=/data
WORKDIR /data
CMD ["fpi_check", "--help"]
