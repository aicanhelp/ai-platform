<!--
**Note:** When your OFEP is complete, all of these comment blocks should be removed.

** 你可以使用中文书写 ofep, 但请将模版中所有的英文层级标题保留。**

To get started with this template:

- [ ] **Make a copy of this template directory.**
  Copy this template into the owning SIG's directory and name it
  `NNNN-short-descriptive-title`, where `NNNN` is the gitee issue number (with no
  leading-zero padding) assigned to your enhancement above.
- [ ] **Fill out this file as best you can.**
  At minimum, you should fill in the "Summary" and "Motivation" sections.
  These should be easy if you've preflighted the idea of the OFEP with the
  appropriate SIG(s).
- [ ] **Create a PR for this OFEP.**
  Assign it to people in the SIG who are sponsoring this process.
- [ ] **Merge early and iterate.**
  Avoid getting hung up on specific details and instead aim to get the goals of
  the OFEP clarified and merged quickly. The best way to do this is to just
  start with the high-level sections and fill out details incrementally in
  subsequent PRs.

When editing OFEPS, aim for tightly-scoped, single-topic PRs to keep discussions
focused. If you disagree with what is already in a document, open a new PR
with suggested changes.

One OFEP corresponds to one "feature" or "enhancement" for its whole lifecycle.

The canonical place for the latest set of instructions (and the likely source
of this file) is [here](/proposals/NNNN-ofep-template/README.md).
-->

# OFEP-NNNN: Your short, descriptive title

<!--
This is the title of your OFEP. Keep it short, simple, and descriptive. A good
title can help communicate what the OFEP is and should be considered as part of
any review.
-->

<!--
A table of contents is helpful for quickly jumping to sections of a OFEP and for
highlighting any additional information provided beyond the standard OFEP
template.
-->

- [OFEP-NNNN: Your short, descriptive title](#ofep-nnnn-your-short-descriptive-title)
  - [Release Signoff Checklist](#release-signoff-checklist)
  - [Authors](#authors)
  - [Summary](#summary)
  - [Motivation](#motivation)
    - [Goals](#goals)
    - [Non-Goals (Optional)](#non-goals-optional)
  - [Proposal](#proposal)
    - [User Stories (Optional)](#user-stories-optional)
      - [Story 1](#story-1)
      - [Story 2](#story-2)
    - [Notes/Constraints/Caveats (Optional)](#notesconstraintscaveats-optional)
    - [Risks and Mitigations (Optional)](#risks-and-mitigations-optional)
  - [Design Details](#design-details)
  - [Impact Analysis](#impact-analysis)
    - [Affected Components](#affected-components)
    - [Dependencies](#dependencies)
  - [Usage](#usage)
    - [User Instructions](#user-instructions)
    - [API Documentation (Optional)](#api-documentation-optional)
  - [Deployment](#deployment)
    - [Deployment Plan](#deployment-plan)
    - [Post-Deployment Verification](#post-deployment-verification)
  - [Test Proposal](#test-proposal)
  - [Resource Consumption Analysis](#resource-consumption-analysis)
  - [Security Considerations](#security-considerations)
  - [Drawbacks](#drawbacks)
  - [Alternatives](#alternatives)

## Release Signoff Checklist

<!--
**ACTION REQUIRED:** In order to merge code into a release, there must be an
issue in [kubesphere/project] referencing this OFEP and targeting a release
milestone **before the `Enhancement Freeze` of the targeted release**.

Check these off as they are completed for the Release Team to track. These
checklist items _must_ be updated for the enhancement to be released.
-->

Items marked with (R) are required *prior to targeting to a milestone / release*.

- [ ] (R) OFEP approvers have approved the OFEP
- [ ] (R) Design details are appropriately documented
- [ ] (R) Test proposal is in place

<!--
**Note:** This checklist is iterative and should be reviewed and updated every time this enhancement is being considered for a milestone.
-->

## Authors
<!--
Github IDs
-->

## Summary

<!--
This section is incredibly important for producing high-quality, user-focused
documentation such as release notes or a development roadmap. It should be
possible to collect this information before implementation begins, in order to
avoid requiring implementors to split their attention between writing release
notes and implementing the feature itself. OFEP editors and SIG Docs
should help to ensure that the tone and content of the `Summary` section is
useful for a wide audience.

A good summary is probably at least a paragraph in length.
-->

## Motivation

<!--
This section is for explicitly listing the motivation, goals, and non-goals of
this OFEP.  Describe why the change is important and the benefits to users. 
-->

### Goals

<!--
List the specific goals of the OFEP. What is it trying to achieve? How will we
know that this has succeeded?
-->

### Non-Goals (Optional)

<!--
What is out of scope for this OFEP? Listing non-goals helps to focus discussion
and make progress.
-->

## Proposal

<!--
This is where we get down to the specifics of what the proposal actually is.
This should have enough detail that reviewers can understand exactly what
you're proposing, but should not include things like API designs or
implementation. What is the desired outcome and how do we measure success?.
The "Design Details" section below is for the real
nitty-gritty.
-->

### User Stories (Optional)

<!--
Detail the things that people will be able to do if this OFEP is implemented.
Include as much detail as possible so that people can understand the "how" of
the system. The goal here is to make this feel real for users without getting
bogged down.
-->

#### Story 1

#### Story 2

### Notes/Constraints/Caveats (Optional)

<!--
What are the caveats to the proposal?
What are some important details that didn't come across above?
Go in to as much detail as necessary here.
This might be a good place to talk about core concepts and how they relate.
-->

### Risks and Mitigations (Optional)

<!--
What are the risks of this proposal, and how do we mitigate? Think broadly.
For example, consider both security and how this will impact the larger
Kubesphere ecosystem.

How will security be reviewed, and by whom?

How will UX be reviewed, and by whom?
-->

## Design Details

<!--
This section should contain enough information that the specifics of your
change are understandable. This may include API specs (though not always
required) or even code snippets. If there's any ambiguity about HOW your
proposal will be implemented, this is the place to discuss them.
-->

## Impact Analysis

### Affected Components

### Dependencies

## Usage

### User Instructions
<!--
Provide instructions on how to use the new feature/function, including screenshots or examples if necessary.

This is NOT a formal user guide documentation, but should at least be enough to tell anyone how to enable and use this feature.

For example:
This feature can be enabled by setting xxx.enabled=true in values.yaml of the extension, install the extension, then login console and go to xxx page and you will see menu yyy.
-->

### API Documentation (Optional)

## Deployment

### Deployment Plan
<!--
Describe the deployment plan, including steps, timeline, and rollback procedures.
-->

### Post-Deployment Verification
<!--
Describe the steps to verify that the deployment was successful and that the feature/function is working as expected.

For example:
Run `kubectl get pod -n xxx`, if you see a pod yyy is running, it proves the new feature is successfully deployed.
-->

## Test Proposal
<!--
Please propose the major test scenarios to the Test Team, more is better.

This is NOT a formal test plan (aka user story test plan which should be designed and executed by the Test Team), the purpose is to highlight the user-facing changes and/or corner cases introduced by this feature to the Test Team, so they could be handled well.
-->

## Resource Consumption Analysis
<!--
Resource consumption analysis including CPU, RAM, disk, IO, PIDs, sockets, inodes etc. based on the feature design or expected code changes from developer perspective.
-->

## Security Considerations

Check the items if this feature applies:

- [ ] Creates nodePort service(s)
- [ ] Creates privileged container(s)
- [ ] Mounts and write root filesystem(including subdirectories) on the node
- [ ] Manages raw network interfaces
- [ ] Manages raw disk devices
- [ ] Requires cluster-admin permission

## Drawbacks

<!--
Why should this OFEP _not_ be implemented?
-->

## Alternatives

<!--
What other approaches did you consider, and why did you rule them out? These do
not need to be as detailed as the proposal, but should include enough
information to express the idea and why it was not acceptable.
-->