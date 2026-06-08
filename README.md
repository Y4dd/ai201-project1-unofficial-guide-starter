# The Unofficial Guide — Project 1

## Domain

<!-- What topic or category of knowledge does your system cover?
     Why is this knowledge valuable, and why is it hard to find through official channels?
     Example: "Student reviews of CS professors at [university] — useful because official
     course descriptions don't reflect teaching style, exam difficulty, or workload." -->

On and Off Campus housing and landlord navigation at Western Michigan University
This knowledge is valuable because:

- Selecting housing is a small choice with year-long consequences due to leasing requirements
- The On-campus official university resources are vague and hard to navigate
- The Off-Campus only showcase polished best-case scenarios
- I had a very bad personal experience finding housing

This RAG should provide a system of clarity aiding in the choice of housing.

---

## Document Sources

<!-- List every source you collected documents from.
     Be specific: include URLs, subreddit names, forum thread titles, or file names.
     Aim for variety — sources that together cover different subtopics or perspectives. -->

| #   | Source                                              | Description                                                                                                              | URL or location                                                           |
| --- | --------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------- |
| 1   | WMU Housing — Residence Halls                       | 2025–26 and 2026–27 rate tables for all 5 halls; room types, amenities, eligibility per hall                             | wmich.edu/housing/info/rates · wmich.edu/housing/options/halls            |
| 2   | WMU Housing — On-Campus Apartments                  | Monthly rates and full cancellation fee schedule for Arcadia Flats, Western View, Stadium Drive, Spindler                | wmich.edu/housing/apartment-rates · wmich.edu/housing/info/cancelcontract |
| 3   | ApartmentRatings.com / BBB — Landlord Profiles      | Review aggregations and BBB records for Concord Place (Edward Rose), Icon Properties, Nu Partners                        | apartmentratings.com · bbb.org                                            |
| 4   | ApartmentRatings.com — Off-Campus Complexes         | 108–164 student reviews each for The Paddock, 58 West, The Wyatt, and Bronco Club                                        | apartmentratings.com · complex websites                                   |
| 5   | Western Herald — Campus Neighborhood Safety         | Four rental zone profiles; crime stats within 2-mile campus radius; Fraternity Village pedestrian safety                 | westernherald.com · fox17online.com                                       |
| 6   | KMetro + WMU Busing & Parking                       | KMetro routes 3/16/21, Bronco Card free bus pass policy, full WMU parking permit pricing table, winter commuting hazards | wmich.edu/busing · wmich.edu/parking · kmetro.com                         |
| 7   | Michigan Legislature — MCL 554.609 & 554.613        | Security Deposit Act (Act 348): 30-day itemization rule, exact statutory language, 2× penalty                            | legislature.mi.gov (sourced via hemlane.com)                              |
| 8   | EnergySage / Numbeo / RentCafe — Utility Benchmarks | Kalamazoo seasonal utility cost ranges, complex-by-complex utility inclusion matrix, drafty-unit risk factors            | energysage.com · numbeo.com · rentcafe.com                                |

> **Note on data collection:** An initial attempt used `collect_documents.py` (a PRAW-based Reddit scraper targeting r/WMU and r/kzoo) to gather student review data for sources 3–6 and 8. Reddit's API blocked all anonymous and script-level access before any data could be collected. The 8 documents above were compiled instead from official WMU pages, ApartmentRatings.com, review aggregators, and primary legal sources via web research. `collect_documents.py` is preserved in the repo as a record of this attempt.

---

## Chunking Strategy

<!-- Describe your chunking approach with enough specificity that someone else could reproduce it.
     Include:
     - Chunk size (characters or tokens) and why that size fits your documents
     - Overlap size and why (or why not) you used overlap
     - Any preprocessing you did before chunking (e.g., stripping HTML, removing headers)
     - What your final chunk count was across all documents -->

**Chunk size:** One `##` or `###` section per chunk (roughly 200–700 characters each).

**Overlap:** None. Each section covers one topic (one hall, one complex, one legal rule) so there's nothing useful to carry over.

**Why these choices fit your documents:** The documents are structured markdown, splitting on headers keeps each section intact as a coherent unit. A fixed character split would cut through pricing tables mid-row, making the retrieved chunk meaningless. Header-based splitting keeps tables whole and matches how a student would actually ask a question.

**Final chunk count:** 128 chunks

---

## Embedding Model

<!-- Name the embedding model you used and explain your choice.
     Then answer: if you were deploying this system for real users and cost wasn't a constraint,
     what tradeoffs would you weigh in choosing a different model?
     Consider: context length limits, multilingual support, accuracy on domain-specific text,
     latency, and local vs. API-hosted. -->

**Model used:** `multi-qa-MiniLM-L6-cos-v1` via sentence-transformers. Same size and speed as the commonly used `all-MiniLM-L6-v2`, but fine-tuned on question-answer pairs rather than general sentence similarity — a better fit since every query in this system is a student question and every chunk is a potential answer.

**Top-k:** 5. Enough to cover multi-part questions.

**Production tradeoff reflection:**

- For a real deployment, the main upgrade would be switching to a larger model like `all-mpnet-base-v2` or a domain-fine-tuned model for better accuracy on housing-specific language.
- A second improvement would be adding MMR (Maximal Marginal Relevance) retrieval — instead of returning the 5 most similar chunks, it alternates between similarity and diversity, preventing 5 near-identical complaint quotes about the same complex from crowding out other relevant sections.
- At larger scale, hybrid retrieval (BM25 keyword search + semantic search combined) would help with exact-match queries.

---

## Grounded Generation

<!-- Explain how your system enforces grounding — how does it prevent the LLM from answering
     beyond the retrieved documents?
     Describe both your system prompt (what instruction you gave the model) and any structural
     choices (e.g., how you formatted the context, whether you filtered low-relevance chunks).
     Do not just say "I told it to use the documents" — show the actual instruction or explain
     the mechanism. -->

**System prompt grounding instruction:**

> You are a housing advisor for Western Michigan University students.
>
> Answer the student's question using ONLY the information provided in the context below.
> Do not use any knowledge from outside these documents.
> If the context does not contain enough information to answer the question, respond with exactly:
> "I don't have enough information in my documents to answer that question."

> At the end of every answer, add a "Sources:" line listing the source filenames you used
> (e.g., "Sources: source1_wmu_residence_halls.md, source6_winter_commuting.md").

Each retrieved chunk is injected into the system prompt instead of the user turn in order to enforce it as an authoritative instruction instead of being interpreted as conversational input.

**How source attribution is surfaced in the response:**

The system prompt instructs the model to append a Sources: line at the end of every answer naming the .md filenames it drew from. This is LLM-generated attribution from the context headers.

Moreover, in the Gradio UI, The "Retrieved from" is populated separately through the chunk's metadata regadless of the LLMs choices.

---

## Evaluation Report

<!-- Run your 5 test questions from planning.md through your system and record the results.
     Be honest — a partially accurate or inaccurate result that you explain well is more
     valuable than a suspiciously perfect result. -->

| #   | Question                                       | Expected answer            | System response (summarized)                                              | Retrieval quality  | Response accuracy  |
| --- | ---------------------------------------------- | -------------------------- | ------------------------------------------------------------------------- | ------------------ | ------------------ |
| 1   | Which is cheaper, On-campus or Off-campus?     | Off-Campus                 | Off-campus, specifically renting from individual landlords                | Relevant           | Accurate           |
| 2   | What is the closest Off-Campus housing option? | The Tate on Howard         | Mentions street and how close but not the name of the appt. cmplx.        | Relevant           | Partially Accurate |
| 3   | How is Off-Campus transportation?              | Mentions Routes + Bus Pass | Explains the routes well, but never mentions fare unless explicitly asked | Partially relevant | Partially Accurate |
| 4   | Does hunter's ridge include utilities in rent? | Yes, except electric bill  | Sources do not answer it, but inferred that it does not                   | Partially relevant | Accurate           |
| 5   | What is the most dangerous place to rent in?   | Near Downtowwn             | Vine District (Downtown)                                                  | Relevant           | Accurate           |

**Retrieval quality:** Relevant / Partially relevant / Off-target  
**Response accuracy:** Accurate / Partially accurate / Inaccurate

---

## Failure Case Analysis

<!-- Identify at least one question where retrieval or generation did not work as expected.
     Write a specific explanation of *why* it failed, tied to a part of the pipeline.

     "The answer was wrong" is not an explanation.

     "The relevant information was split across a chunk boundary, so retrieval returned
     only half the context — the model didn't have enough to answer correctly" is an explanation.

     "The embedding model treated the professor's nickname as out-of-vocabulary and returned
     results from an unrelated review" is an explanation. -->

**Question that failed:** How is Off-Campus transportation?

**What the system returned:** Off-campus transportation includes WMU-operated campus shuttles and KMetro routes. The WMU-operated shuttles have routes such as Ring Road (#19), Parkview (#25), and the Aviation Shuttle. The KMetro routes that serve the WMU campus include Route 3, Route 16, and Route 21, which stop directly at the WMU campus loading zone. Some apartment complexes, like those on West Main/Drake and the Howard/Kendall corridor, have direct, no-transfer rides to campus via Route 3 or Route 21.

**Root cause (tied to a specific pipeline stage):** Retrieval problem, the question is vague and is less semantically close to the source "KMetro Bus Pass", words like "Price", "Free", "Cost" are not part of the question. i debugged this and found that the chunk containing this is ranked as 7th. I've a test under [tests/eval_retrieve.py](./tests/eval_retrieve.py) with K=9 that retrieves succesfully.

**What you would change to fix it:**
Metadata such as apt. complex name should be taken in consideration when fetching, i would also implement better retreival solutions such as query expansions through the LLM or doing a hybrid search to include semantic matches with cosine similarity since some supplemental information (such as the free bus pass) are not included.

---

## Spec Reflection

<!-- Reflect on how planning.md shaped your implementation.
     Answer both questions with at least 2–3 sentences each. -->

**One way the spec helped you during implementation:** Specs helped ground the planning and implementation tasks.

**One way your implementation diverged from the spec, and why:** I created specs for each phase of the implementation, while the purpose was to have a spec per milestone, i accidentally was vague in prompting and had one planning spec implement both Chunking and Embedding, I didn't like this because this lowers the context per task ratio and could possibly hallucinate or miss things, Moreover, it makes it harder to modify or extend the implementation if concerns are mixed.

---

## AI Usage

<!-- Describe at least 2 specific instances where you used an AI tool during this project.
     For each: what did you give the AI as input, what did it produce, and what did you
     change, override, or direct differently?

     "I used Claude to help me code" is not sufficient.
     "I gave Claude my Chunking Strategy section from planning.md and asked it to implement
     chunk_text(). It returned a function using a fixed character split. I overrode the
     chunk size from 500 to 200 because my documents are short reviews, not long guides." -->

**Instance 1**

- _What I gave the AI:_ I used planning mode with [superpowers:writing-plans](https://github.com/obra/superpowers) given the planning.md requirements for each Milestone
- _What it produced:_ Full design and implementation plan for [ingestion](./docs/superpowers/plans/2026-06-07-ingest-pipeline.md) and other milestones
- _What I changed or overrode:_ Split the implementation afterward to separate commits by milestones, since I messed up in properly separating the concern.

**Instance 2**

- _What I gave the AI:_ A curated list of the resources and asked it to use Web Search to query them and summarize them in .md format under `/documents`, then run subagents to reread the material separately and verify the content
- _What it produced:_ A list of markdown files, each file for a source, properly formatted for chunking.
- _What I changed or overrode:_ It missed some information from the sources even after running verification, which I had to explicitly point it to fetch. This was a major data-integrity red-flag.
