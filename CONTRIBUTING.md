## Contributing Bug Reports

MobileCoin is a prototype that is being actively developed.

`mc-oblivious` was created to meet the needs of *MobileCoin Fog*, to be used in SGX enclaves in fog servers.

`mc-oblivious` is, as far as we know, the *first* oblivious RAM implementation world-wide
that will deployed to production and serve real user requests.

Please report issues to https://github.com/mobilecoinofficial/mc-oblivious/issues.

1. Search both open and closed tickets to make sure your bug report is not a duplicate.
1. Do not use github issues as a forum. To participate in community discussions, please use the community forum at [community.mobilecoin.foundation](https://community.mobilecoin.foundation).

## Pull Requests (PRs)

Pull requests are welcome!

Oblivious RAM is complex and we are actively working to improve the quality of our implementation.
There are many exciting avenues of research especially to improve performance.

If you plan to open a pull request, please install the code formatting git hooks, so that your code will be formatted when you open the PR.

Also, note the `.cargo/config` file which sets `target-cpu=skylake`. If you have an older CPU you may need to change or override this
when you develop locally.

Coming soon: Code Style Guidelines

### Sign the Contributor License Agreement (CLA)

You will need to sign [our CLA](./CLA.md) before your pull request can be merged. Please email [cla@mobilecoin.com](mailto://cla@mobilecoin.com) and we will send you a copy.

## Get in Touch

We're friendly. Feel free to [ping us](mailto://oram@mobilecoin.com)!
