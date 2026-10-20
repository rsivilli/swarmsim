# Security Policy

## Supported Versions

swarmsim is pre-1.0. Security fixes land on `main` and in the next release; older releases are not patched.

## Reporting a Vulnerability

Please **do not** open a public issue for security problems.

Report vulnerabilities privately through GitHub's
[private vulnerability reporting](https://github.com/rsivilli/swarmsim/security/advisories/new)
(the **Security** tab → **Report a vulnerability**). Include:

- A description of the issue and its impact
- Steps or a minimal scenario/config to reproduce it
- The affected version or commit

You should get an acknowledgement within a week. Once a fix is available we'll publish an advisory
and credit you, unless you'd rather stay anonymous.

## Scope

In scope: the `modsim` package and CLI, the Locust replay script, the Dockerfile and the Helm chart
in this repository.

Out of scope: the third-party services the load tester can target (Zenoh, UDL or your own
backend), and findings that require an attacker to already control the scenario files or
timelines you choose to run.
