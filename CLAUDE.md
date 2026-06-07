# Project Context — WMU Unofficial Housing Guide (RAG System)

## What This Project Is

A RAG (Retrieval-Augmented Generation) system that answers student questions about WMU housing — both on-campus and off-campus — using real documents as the knowledge base. The system should surface information that students can't easily find through official channels: actual utility costs, landlord reputation, neighborhood safety, lease traps, and legal rights.

**Stack:** sentence-transformers (embeddings) · ChromaDB (vector store) · Groq (LLM) · python-dotenv

---

## Session Progress Summary

### What's Done

**All 8 source documents are written and verified in `documents/`.**

Each was generated from live web sources, then verified by parallel subagents that re-fetched sources and corrected inaccuracies. Do not regenerate these files — they are authoritative.

| File | Content | Status |
|------|---------|--------|
| `source1_wmu_residence_halls.md` | Every hall (Henry, Valley, Western Heights, Valley Oaks, Britton/Hadley) with 2025–26 and 2026–27 full rate tables, room types, amenities | ✅ Verified & corrected |
| `source2_wmu_apartment_policies.md` | Arcadia Flats, Western View, Stadium Drive, Spindler — monthly rates, full cancellation fee schedule by date | ✅ Verified & corrected |
| `source3_landlord_watchlist.md` | Edward Rose/Concord Place, Icon Properties, Nu Partners — reputation, review data, red flags | ✅ Verified & corrected |
| `source4_offcampus_complexes.md` | The Paddock, 58 West, The Wyatt, Bronco Club — reviews, amenities, management issues, utility inclusion | ✅ Verified & corrected |
| `source5_neighborhood_geography.md` | Four rental zones mapped: Vine, West Main/Drake, Howard/Kendall, Fraternity Village — safety stats, distances, character | ✅ Verified & corrected |
| `source6_winter_commuting.md` | Bronco Card free bus pass, KMetro routes 3/16/21, WMU shuttles, full parking permit pricing, winter driving hazards | ✅ Verified & corrected |
| `source7_michigan_tenant_law.md` | Michigan Security Deposit Act (Act 348) — 30-day rule, exact statutory bold-text language, 2× penalty, MCL citations | ✅ Verified & corrected (legal text precision-checked against MCL) |
| `source8_utility_costs.md` | Seasonal heating/electric cost ranges, which complexes include utilities, drafty-unit risk factors | ✅ Verified & corrected |

**Key corrections made during verification (don't revert these):**
- Valley Oaks bathrooms are community (3–6 residents), not private per room
- Western Heights is first-year only (no East/West split in eligibility)
- Western Heights has AC (comparison chart confirms Yes)
- The Wyatt is ~4 miles from WMU, not ~2 miles
- The 2015 fatal shooting was at Copper Beech Townhomes (2804 W Michigan Ave), not The Bronco Club — different complex, different address
- Michigan tenant law bold-text quote uses "7 days" (numeral), "receipt of same" (no "the"), and a comma before "otherwise" — per MCL 554.609 verbatim
- Penalty basis is "security deposit retained" (MCL 554.613), not "amount wrongfully withheld"
- WMU closure stat: 20+ times since 1999 (not "16 times over 18 years")
- After-hours parking: Mon–Thu after 5 PM; Fridays after 4 PM; weekends free

### What's NOT Done Yet

**`planning.md` is partially filled.** Documents table ✅ and Chunking Strategy ✅ are done. Still needed:
- Domain statement
- Retrieval approach (embedding model, top-k, production reflection)
- 5 test questions with expected answers
- Anticipated challenges
- Architecture diagram
- AI tool plan per milestone

**Pipeline code hasn't been written.** The three remaining milestones:

- **Milestone 3 — Ingestion & Chunking:** Write `ingest.py` that reads `documents/*.md`, chunks them, and loads into ChromaDB. **Chunking strategy decided:** split on `##`/`###` header boundaries; keep each section as one chunk (target 200–700 chars); never split a line starting with `|` (table rows); drop chunks < 80 chars. No overlap between sections.
- **Milestone 4 — Embedding & Retrieval:** Configure sentence-transformers (`all-MiniLM-L6-v2`) to embed chunks into ChromaDB. Write `retrieve.py` with a query function returning top-k chunks.
- **Milestone 5 — Generation & Interface:** Wire ChromaDB retrieval into Groq (llama-3 model). Add a grounding system prompt. Build a Gradio or Streamlit interface. The system prompt must instruct the model to only answer from retrieved context and cite sources.

---

## Key Technical Decisions

**Why these document topics:** The 8 sources cover the full decision surface for a WMU student choosing housing — official baselines (sources 1–2), unfiltered reputation data (sources 3–4), location context (source 5), practical logistics (source 6), legal protection (source 7), and hidden costs (source 8).

**Data sources used (for planning.md and README.md):**
- wmich.edu/housing (official WMU housing pages)
- wmich.edu/housing/info/rates, wmich.edu/housing/apartment-rates
- wmich.edu/housing/info/cancelcontract
- wmich.edu/busing, wmich.edu/parking/permit-pricing
- ApartmentRatings.com (108–155 reviews per complex)
- Yelp, BBB, complex official websites
- hemlane.com/resources/michigan-security-deposit-laws + Michigan Legislature (MCL 554.609, 554.613)
- EnergySage, Numbeo, RentCafe for utility benchmarks
- Western Herald (westernherald.com) for campus safety article
- Fox17 for Fraternity Village Drive pedestrian safety

**Reddit / collect_documents.py:** An initial attempt used a PRAW-based scraper (`collect_documents.py`) targeting r/WMU and r/kzoo. Reddit blocked all script-level API access before any data was collected. The script is gutted to a stub and kept only as a record of the attempt; `praw` has been removed from `requirements.txt`. The 8 documents cover the same information through other sources.

---

## Suggested Next Session Start

1. Finish `planning.md` — domain statement, retrieval approach, 5 test questions, anticipated challenges, architecture diagram, AI tool plan
2. Write `ingest.py` (Milestone 3) — chunk the 8 markdown files on `##`/`###` headers and load into ChromaDB
3. Write `retrieve.py` (Milestone 4) — embed queries and return top-k chunks
4. Write `generate.py` + interface (Milestone 5) — Groq + grounding prompt + Gradio/Streamlit

Good test questions to use (based on the documents):
1. "What are the cancellation fees if I sign a lease at Arcadia Flats and need to back out in April?"
2. "What do students say about The Wyatt's noise and security?"
3. "If my landlord doesn't return my security deposit after 30 days, what can I do under Michigan law?"
4. "Which KMetro bus routes stop at the WMU campus, and do I need to pay?"
5. "What is the typical winter heating bill for a Kalamazoo apartment?"
