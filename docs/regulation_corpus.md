# Fetching the regulation corpus

`docai/` cites regulations in the compliance reports it drafts. The source
documents are the official texts, fetched from the issuing authority — **not
committed to this repo**. `docai/regulations_corpus/` is gitignored: these are
third-party Government of India documents, they total 8.6 MB, and they are
reproducible from the URLs below.

| document | version / as on | official source |
|---|---|---|
| Master Direction — Know Your Customer (KYC) Direction, 2016 (RBI/DBR/2015-16/18) | issued 25 Feb 2016, **updated as on 14 Aug 2025** | [rbi.org.in notification 2607](https://www.rbi.org.in/CommonPerson/english/scripts/notification.aspx?id=2607) → [PDF](https://rbidocs.rbi.org.in/rdocs/notification/PDFs/MD18KYCF6E92C82E1E1419D87323E3869BC9F13.PDF) |
| The Prevention of Money-Laundering Act, 2002 | Act 15 of 2003, as amended | [fiuindia.gov.in](https://fiuindia.gov.in/files/AML_Legislation/pmla_2002.html) → [PDF](https://fiuindia.gov.in/pdfs/AML_legislation/PMLA_Act_2002.pdf) |
| Prevention of Money-laundering (Maintenance of Records) Rules, 2005 | as amended to the **Third Amendment Rules, 2023** | [fiuindia.gov.in](https://fiuindia.gov.in/files/AML_Legislation/notification.html) → [PDF](https://fiuindia.gov.in/pdfs/AML_legislation/PMLA_2005.pdf) |
| FIU-IND Document Reporting Format (CTR / STR / CCR / NTR, and their electronic file structures) | **Version 1.14** | [fiuindia.gov.in downloads](https://fiuindia.gov.in/files/Downloads/Downloads.html) → [PDF](https://fiuindia.gov.in/pdfs/quicklinks/Reporting_Format.pdf) |

To fetch them:

```bash
mkdir -p docai/regulations_corpus && cd docai/regulations_corpus
curl -L -o pmla_act_2002.pdf           https://fiuindia.gov.in/pdfs/AML_legislation/PMLA_Act_2002.pdf
curl -L -o pml_rules_2005.pdf          https://fiuindia.gov.in/pdfs/AML_legislation/PMLA_2005.pdf
curl -L -o fiu_ind_reporting_format.pdf https://fiuindia.gov.in/pdfs/quicklinks/Reporting_Format.pdf
```

**The RBI PDF needs a warmed session** — `rbidocs.rbi.org.in` closes the
connection on a cold programmatic request. Fetch the notification page first,
keep the cookie, then request the PDF with a `Referer`:

```bash
curl -L -c /tmp/rbi.jar -A "Mozilla/5.0" \
  "https://www.rbi.org.in/Scripts/BS_ViewMasDirections.aspx?id=11566" -o /dev/null
curl -L -b /tmp/rbi.jar -A "Mozilla/5.0" \
  -e "https://www.rbi.org.in/Scripts/BS_ViewMasDirections.aspx?id=11566" \
  -o rbi_kyc_master_direction.pdf \
  "https://rbidocs.rbi.org.in/rdocs/notification/PDFs/MD18KYCF6E92C82E1E1419D87323E3869BC9F13.PDF"
```

Verify every download is a PDF and not a block page — a bot-block returns
HTTP 200 with HTML, which is the failure most likely to go unnoticed:

```bash
for f in *.pdf; do [ "$(head -c 4 "$f")" = "%PDF" ] && echo "ok $f" || echo "NOT A PDF: $f"; done
```

Sizes as fetched on 27 Sep 2026: RBI KYC 2,049,883 B (~214 pp), PMLA
504,120 B (~47 pp), PML Rules 335,649 B (~25 pp), FIU-IND Reporting Format
5,978,084 B (~539 pp).

**These four PDFs are the live retrieval index.** They are imported into a
Vertex AI Search data store (`mulenet-aml-regulations`, `global`), and with
`MULENET_DOCAI_MODE=cloud` the report agent cites real passages from them —
title, page and the authority's URL, never a composed string.

A citation is verified by **retrieval identity**: `search()` records every
citation it returns, and `verify_citation()` checks that ledger. A citation
that retrieval never produced cannot be verified, so
`"PMLA, 2002, p. 9999"` fails even though the PMLA is genuinely indexed. The
default mode stays placeholder-backed so the test suite runs offline and
free, and those placeholders are loudly labelled `SYNTHETIC — NOT A REAL
CITATION`.
