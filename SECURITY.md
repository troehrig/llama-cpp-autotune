# Security Policy

## Supported versions

Security fixes are applied to the latest released version. During the initial
public phase, this means the `0.19.x` line.

## Reporting a vulnerability

Please use GitHub's private vulnerability reporting feature for this
repository. Include the affected version, impact, reproduction conditions, and
a minimal proof of concept when appropriate.

If private reporting is unavailable, open a public issue containing only a
request for a private contact channel. Do not publish exploit details,
credentials, private model responses, network addresses, or unreviewed run
artifacts.

## Network exposure

Generated deployment commands bind to `127.0.0.1` by default. Supplying
`--deployment-host 0.0.0.0` intentionally exposes the generated server command
on all interfaces. The operator is responsible for firewalling, authentication,
TLS termination, access control, and the security of the surrounding network.

Autotune is a benchmarking and configuration aid, not a hardened API gateway.
