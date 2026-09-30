# Extraction prompt — paste this once at the start of a chat session

Then paste one `batches/batch-XXX.txt` per turn. Reply to each with JSON only.

---

You extract evidence-grounded events and relations from Indonesian news for the
East Java Food-System Temporal Event Knowledge Graph. This is LLM-only silver
extraction, not human verification.

Return **only** a JSON array, one object per article, in the order given. No
prose, no markdown fences, no commentary.

## Output schema

```json
[
  {
    "candidate_id": "ann_xxxxxxxx",
    "article_relevance": "yes | no | uncertain",
    "east_java_basis": "specific_occurrence | named_participant | measured_inclusion | list_mention | venue_only | none",
    "article_confidence_pct": 0,
    "events": [
      {
        "event_slot": 1,
        "event_kind": "disruption | response",
        "event_subtype": "…",
        "event_date": "YYYY-MM-DD",
        "event_end_date": "",
        "date_precision": "day | estimated_day | month | year | range",
        "location": "…",
        "commodity": "…",
        "actor": "…",
        "food_system_stage": "production | distribution | processing | retail | consumption | food_safety",
        "evidence_quote": "exact substring of the article body",
        "confidence_pct": 0,
        "granularity_basis": "incident | episode_member"
      }
    ],
    "relations": [
      {
        "relation_slot": 1,
        "subject_event_slot": 1,
        "predicate": "respondsTo | reportedCauseOf | implementsPolicy | citesPolicy",
        "object_event_slot": 2,
        "policy_id": "",
        "evidence_quote": "exact substring of the article body",
        "confidence_pct": 0
      }
    ]
  }
]
```

## Core rules

- Extract only **concrete observed** East Java food-system occurrences. A
  national occurrence counts only when the article states an East Java-specific
  fact about it.
- **A price statement alone is not a disruption.** A disruption requires explicit
  shortage, damage, crop or production failure, distribution failure,
  food-safety harm, disaster impact, pest, or disease.
- A **response** is an action already started, operational, or completed by the
  publication date. Exclude forecasts, requests, promises and unimplemented
  plans. Evidence saying only *akan, berencana, diharapkan, berpotensi,
  mengimbau, meminta, menyarankan, sedang dikaji* is not an implemented response.
- A response may exist without a disruption. A `respondsTo` **relation** requires
  an explicit textual connection; co-occurrence is not enough.
- At most **three** events per article. Do not split one programme into one event
  per sentence.
- Every `evidence_quote` must be an **exact substring** of that article's body,
  copied character for character.

## Subtypes

- disruption: `WeatherDisruption`, `DisasterEvent`, `PestDiseaseEvent`,
  `ProductionDisruption`, `SupplyShortage`, `DistributionDisruption`,
  `FoodSafetyIncident`
- response: `ReserveRelease`, `FoodAssistance`, `MarketOperationAction`,
  `ProcurementAction`, `DistributionIntervention`, `ImportExportAction`,
  `InspectionEnforcement`, `ProductionAssistance`, `CrisisPreparednessAction`,
  `OtherResponseAction`

Floods, landslides, earthquakes and eruptions are `DisasterEvent` under
`event_kind: disruption`. Staple packages already delivered to affected
residents are `FoodAssistance`. Monitoring, pumps and deployed emergency
equipment are `CrisisPreparednessAction`. Never pair `event_kind: response` with
a disruption subtype or vice versa.

## Commodity

Use the commodity named for that occurrence; never invent one. `milk` for
susu (never `sugar`). Use `broad_food` for mixed or general food, `unclear` when
no food commodity is stated. Others: `rice`, `chili`, `shallot`, `garlic`,
`beef`, `buffalo_meat`, `chicken_meat`, `chicken_eggs`, `sugar`, `cooking_oil`,
`soybean`, `corn`, `wheat`, `fish`.

## Dates

- Use an explicitly stated date when available; otherwise set `event_date` to the
  publication date with `date_precision: estimated_day`.
- Month precision → last day of that month. Year precision → 1 January.
- `event_end_date` is `""` unless the article supports a range.
- **No event date may be later than the publication date.**

## Granularity — read this carefully

Extract at **incident** level: one datable occurrence with its own actors and
location.

A sustained condition — a months-long shortage, an outbreak wave, a multi-year
programme — is **not** one incident. Extract the specific occurrences the article
reports and set `granularity_basis: "episode_member"`. Never merge separate
reports of a sustained condition into a single event.

## East Java scope — do not answer yes/no alone

Set `east_java_basis`:

| value | meaning |
|---|---|
| `specific_occurrence` | an occurrence located, dated or measured in East Java |
| `named_participant` | an East Java actor or facility operationally involved |
| `measured_inclusion` | a national figure that reports an East Java value |
| `list_mention` | East Java appears in a list of covered provinces |
| `venue_only` | East Java is only where an announcement happened |
| `none` | no East Java link |

A national programme announced in Surabaya is `venue_only`. A national scheme
listing East Java among seven provinces is `list_mention`. Neither is a reason to
omit the occurrence — record it with the correct basis and let the downstream
filter decide.

### These are TWO INDEPENDENT questions. Do not collapse them.

* `east_java_basis` answers **where** — is there an East Java link, and of what
  kind? It is about geography only.
* `article_relevance` answers **what** — does the article report a food-system
  disruption or response? It is about subject matter only.

An article can be strongly East Java and completely irrelevant, or nationally
framed and highly relevant. All four combinations occur:

| Article | basis | relevance | why |
|---|---|---|---|
| "Gempa M 4,2 Terjadi di Jember" | `specific_occurrence` | `no` | dated in Jember, but an earthquake report with no food-system impact stated |
| "8 Cara Menyelamatkan Diri dari Gempa" (mentions Jember) | `specific_occurrence` | `no` | a safety how-to, not an occurrence |
| "Banjir Lumpur Terjang Permukiman Warga Jember" | `specific_occurrence` | `yes` | a dated East Java disaster affecting residents |
| "Bapanas salurkan beras ke 7 provinsi termasuk Jatim" | `list_mention` | `yes` | food-system action, thin East Java link |
| "KKP amankan kapal ilegal di Selat Malaka" | `none` | `no` | no East Java link at all |

**Never set `east_java_basis: "none"` when the article names an East Java city,
regency or the province itself.** If the place is named, the basis is one of the
other five values, whatever you decide about relevance. Use `none` only when no
East Java location appears anywhere in the article.

## Confidence

Give an evidence-based estimate from 1 to 100 for every emitted event and
relation. Do not reuse one value for everything. Use
`article_confidence_pct: 0` only when the article is irrelevant and no events are
emitted.

## Policy relations

Emit `implementsPolicy` or `citesPolicy` **only** when the article names a
specific regulation (e.g. "UU No. 18 Tahun 2012", "Perpres 125 tahun 2022"). Put
that citation verbatim in `policy_id` and set `object_event_slot: 0`. If no
regulation is named, emit neither.

---

Confirm you have understood by replying `READY`. I will then paste batches one at
a time; reply to each with the JSON array only.
