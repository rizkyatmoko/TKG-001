# FoodJatim TKG ontology v0.4

Version 0.4.0 extends the v0.3 event/response/policy ontology with the
claim-level structures needed for provenance-calibrated temporal GNN research.
It is an additive migration: existing event, policy, evidence, relation, and
direct convenience-edge terms remain available.

## Temporal semantics

The graph separates three times that must not be conflated:

1. **Event valid time** — fj:hasEventTime points to an
   fj:EventTimeInterval represented with OWL-Time. fj:eventDate and
   fj:eventEndDate remain scalar shortcuts for compatibility.
2. **Source report time** — fj:publicationDateTime is the timestamp recorded
   for a news article. fj:reportedAt propagates that timestamp to its event
   mentions, evidence statements, and source relation assertions.
3. **Corpus observation time** — fj:observationDate remains the earliest date
   on which an aggregated entity or assertion became available in the included
   corpus. It is not event time.

For 47 articles without a source time of day, midnight Asia/Jakarta is stored
with report-time precision “day”. The remaining 253 articles use “minute”.

Canonical events carry fj:firstReportedAt and fj:lastReportedAt. Event mentions
also carry fj:reportingLagFromEventStartDays, defined as publication date minus
normalized event start date. For long-running events, this is not a delay from
event completion.

## Publisher provenance

fj:Publisher is a prov:Organization. Every article has one fj:publishedBy
publisher and retains its channel-level database identifier with
fj:sourceChannelKey.

| Source channel keys | Normalized publisher |
|---|---|
| beritajatim | beritajatim |
| bisnis | bisnis |
| detik_finance, detik_food, detik_news | detik |
| kompas_money, kompas_nasional, kompas_regional | kompas |
| kontan | kontan |

Source-derived entities use:

- fj:reportedIn for the article;
- fj:sourcePublisher for the normalized publisher; and
- fj:reportedAt for the source report timestamp.

fj:sourcePublisher is deliberately not an attribution of the LLM extraction
activity to the publisher.

## Confidence semantics

fj:extractionConfidence measures confidence that source content was extracted
and normalized as represented. fj:assertionConfidence measures confidence that
a relation is supported under the annotation procedure. Neither is
automatically a calibrated probability of real-world truth.

The current source JSON supplies one relation-confidence value. For a
backward-compatible v0.4 migration, that value populates both properties and
the limitation is recorded with fj:confidenceBasis. All labels remain LLM-only
silver data with fj:humanVerified false.

## Soft event resolution

The direct EventMention → denotesCanonicalEvent → CanonicalEvent edge remains
for simple queries. Every accepted edge now has an accompanying
fj:EventResolutionAssertion containing:

- fj:resolutionMention;
- fj:resolutionCandidate;
- fj:sameEventProbability;
- fj:resolutionDecision “same_event”;
- fj:resolutionDecisionConfidence;
- fj:resolutionMethod;
- fj:resolutionRationale; and
- article, publisher, report-time, generation, and silver-label provenance.

Rejected candidate pairs are retained separately as
fj:EventIdentityComparison nodes. Their fj:sameEventProbability is the
complement of the LLM's confidence in the distinct_events decision.
Distinct-event comparisons are not source contradictions.

## Corroboration and contradiction

fj:EvidenceAgreementAssertion reifies a comparison between two
fj:RelationAssertion nodes. Its type is either fj:Corroboration or
fj:Contradiction, with confidence, method, provenance, and
fj:independenceAssessed.

The current graph materializes corroboration only when two direct,
evidence-bearing source assertions support the same resolved relation.
fj:independenceAssessed false is explicit because copying, syndication, and
common-statement dependence have not been adjudicated.

No contradiction edge is materialized from the current positive-only
annotations. Absence of a relation, a rejected event merge, temporal distance,
or different wording is insufficient evidence of contradiction.

## Current materialization

The resolved 300-article graph contains:

- 37,821 RDF triples;
- 359 event mentions and 16 canonical events;
- 375 explicit event-time intervals;
- 5 normalized publisher nodes covering 9 source channels;
- 42 soft event-resolution assertions;
- 5 rejected event-identity comparisons;
- 223 evidence-bearing relation assertions;
- 34 evidence-agreement assertions and 34 corroboration edges; and
- 0 contradiction edges.

The graph remains fully LLM-only silver data and excludes price prediction.

## Migration from v0.3

- Existing eventDate, eventEndDate, publicationDateTime,
  extractionConfidence, and denotesCanonicalEvent triples remain.
- Consumers that only use v0.3 direct edges require no query changes.
- GNN exports should prefer hasEventTime, claim-level report time, publisher
  nodes, and reified resolution/agreement assertions.
- owl:sameAs remains intentionally absent because event identity is
  confidence-bearing and method-dependent.
- resolutionMethod no longer has a class-specific RDFS domain, so it can be
  used on canonical events, soft resolution assertions, and pairwise identity
  comparisons without unintended type inference.

## Validation

Use the isolated environment recorded in requirements.txt:

~~~powershell
.\.venv\Scripts\python.exe scripts\build_llm_silver_graph.py
.\.venv\Scripts\python.exe scripts\build_resolved_silver_graph.py
.\.venv\Scripts\python.exe scripts\validate_graph.py --data data\derived\llm-silver-tkg-v2.ttl --require-pyshacl --report data\derived\llm-silver-tkg-v2-validation.json
.\.venv\Scripts\python.exe scripts\verify_resolved_silver_graph.py
~~~

The validator checks time-interval consistency, report-time propagation,
publisher identity, confidence ranges, soft-resolution/direct-edge alignment,
agreement/direct-edge alignment, and the disjoint interpretation of
corroboration and contradiction.

## GNN-facing interpretation

For the future heterogeneous GNN:

- event mentions, canonical events, assertions, articles, publishers, evidence,
  policies, actors, locations, commodities, and stages are distinct node types;
- same-event probability, extraction confidence, assertion confidence, and
  agreement confidence are edge or assertion-node weights;
- event valid time and report time are separate temporal encodings;
- publisher-family holdouts use normalized publisher nodes;
- target respondsTo and implementsPolicy edges must be masked from message
  passing for each query; and
- relation-evidence text that directly reveals a target must not be included in
  target prediction features.

