# Instagram Data Cleaning Report: Kuala Lumpur Standard Chartered Marathon

## 1. Purpose and analytical context

This report documents the preparation of Instagram posts collected for a sentiment analysis of the Kuala Lumpur Standard Chartered Marathon (KLSCM). The source datasets represent posts retrieved using the hashtags `#klscm2019`, `#klscm2023`, `#klscm2024`, and `#klscm2025`. The objective of the cleaning process was to transform four independently collected JSON exports into one consistent, analysis-ready CSV dataset while preserving the text and essential post metadata.

The workflow was implemented in `data_cleaning.ipynb` using pandas. Each major operation was placed in a separate notebook cell so that the process is transparent, reproducible, and easy to audit.

## 2. Source data

The four source files were collected on 10 July 2026 and stored in `Instagram/Raw Data`:

| Dataset label | Source file | Raw records |
|---|---|---:|
| KLSCM 2019 | `2026-07-10_klscm2019.json` | 4,463 |
| KLSCM 2023 | `2026-07-10_klscm2023.json` | 3,578 |
| KLSCM 2024 | `2026-07-10_klscm2024.json` | 3,159 |
| KLSCM 2025 | `2026-07-10_klscm2025.json` | 3,391 |
| **Total** |  | **14,591** |

The dataset label describes the hashtag used for retrieval, not necessarily the publication year of every post. Instagram users may reuse historical hashtags in later posts, publish throwbacks, or mention multiple event years in one caption. This is visible in the timestamp ranges:

| Source hashtag | Earliest timestamp (UTC) | Latest timestamp (UTC) |
|---|---|---|
| `klscm2019` | 2019-02-03 | 2025-10-20 |
| `klscm2023` | 2022-11-12 | 2025-10-04 |
| `klscm2024` | 2023-05-01 | 2025-10-04 |
| `klscm2025` | 2024-08-21 | 2026-05-11 |

Consequently, comparisons in later sentiment analysis should be described as comparisons between **hashtag-defined corpora** unless an additional publication-date filter is applied.

## 3. Schema assessment and standardization

The first object in every JSON file was inspected before transformation. This was necessary because the exports use two schemas:

| Standard field | 2019 and 2023 source | 2024 and 2025 source |
|---|---|---|
| `author_id` | `author.id` | `ownerId` |
| `post_url` | `url` | `url` |
| `caption` | `caption` | `caption` |
| `hashtag` | Assigned from source file | Assigned from source file |
| `timestamp` | `taken_at_timestamp` | `timestamp` |

Standardizing field names was required before concatenation. Without this step, equivalent information would occupy different columns and produce missing values after combining the files. Only five variables were retained because they directly support record identification, traceability, grouping, and text-based sentiment analysis:

- `author_id` identifies the account and supports duplicate detection without retaining the display name.
- `post_url` provides traceability to the original post where it remains accessible.
- `caption` is the principal text input for sentiment analysis.
- `hashtag` records the source corpus or retrieval stratum.
- `timestamp` supports chronological analysis and temporal validation.

Other scraped fields, such as image URLs, media dimensions, likes, comments, profile pictures, and location details, were excluded because they were outside the stated text-sentiment objective and would unnecessarily increase dataset complexity.

## 4. Cleaning methodology and justification

### 4.1 Dependency and path configuration

Pandas was used because it provides consistent functions for reading JSON, selecting and renaming fields, detecting missing values, removing duplicates, concatenating tables, parsing timestamps, and exporting CSV files. Paths were defined with `pathlib.Path` rather than hard-coded operating-system strings. This improves portability and reduces path-separator errors.

The notebook checks that all four source files exist before processing. Failing early is preferable to silently producing a partial combined dataset when an input file is missing.

### 4.2 Reading each JSON file separately

Each file was loaded into its own dataframe. Separate loading and inspection cells preserve the identity of each collection and make it possible to verify raw counts and schema-specific fields before transformation. The first record is displayed as a structural check rather than as substantive analysis.

### 4.3 Caption trimming

Leading and trailing whitespace was removed from captions with string trimming. Such whitespace has no semantic value but can cause visually identical captions to be treated as different during duplicate detection. Internal whitespace, capitalization, punctuation, emoji, spelling, and wording were left unchanged to avoid altering the original expression that will be analyzed for sentiment.

### 4.4 Removing blank captions

Records with null or empty captions were removed because they contain no text for a caption-based sentiment model. Retaining them would increase the reported sample size without contributing analyzable observations and could introduce null-handling errors later. Three records were removed for this reason: two from the 2019 corpus and one from the 2023 corpus.

### 4.5 Within-year duplicate removal

Duplicates were defined using the combination of `author_id` and cleaned `caption`. This rule targets repeated collection of the same textual contribution from the same account while allowing different users to post identical text and allowing one author to publish genuinely different captions.

The combination was selected instead of caption alone because common promotional phrases can legitimately appear across accounts. It was also selected instead of URL alone because the research concern is duplication of author-level textual content, and scraper results can contain repeated content under records that should not be counted twice for sentiment measurement. The first occurrence was retained.

### 4.6 Hashtag assignment

A constant source hashtag was assigned to every row according to its input file. For example, rows originating from the 2019 file receive `klscm2019`. This is more reliable as a corpus label than extracting one value from the post's full hashtag list, since captions may contain multiple KLSCM years and unrelated hashtags.

The field should therefore be interpreted as **source hashtag**, not as a claim that the post was published during that event year or contains only that hashtag.

### 4.7 Data type normalization

Author IDs were converted to strings. Although IDs contain digits, they are identifiers rather than quantities and should not be used in arithmetic. String storage also avoids accidental numeric formatting or inconsistent integer types across source schemas.

Timestamps were parsed as UTC-aware datetime values with invalid values coerced to missing values. UTC provides one consistent time reference across all records. A validation assertion subsequently confirmed that every retained timestamp was successfully parsed.

### 4.8 Combination and cross-year deduplication

The four cleaned dataframes were concatenated row-wise after they shared the same column names and meanings. A second duplicate check using `author_id` and `caption` was then performed across the complete dataset. This was necessary because the same post or caption can be retrieved through more than one KLSCM hashtag. Seventeen additional duplicate records were removed at this stage.

Because the first occurrence is retained and files are concatenated in the order 2019, 2023, 2024, and 2025, a duplicated author-caption pair is assigned to the earliest source corpus in that processing order. This precedence rule should be considered if source-level comparisons are sensitive to posts containing multiple year hashtags.

The final records were sorted by timestamp to provide a stable chronological order. Sorting does not change the observations or their sentiment; it only improves readability and makes temporal inspection easier.

### 4.9 Export format and encoding

The cleaned dataframe was exported as `instagram_cleanded.csv` with the dataframe index excluded. The index is an internal pandas row label and has no analytical meaning, so exporting it would create an unnecessary column.

UTF-8 with a byte-order mark (`utf-8-sig`) was used to improve compatibility with spreadsheet software while preserving multilingual captions, accented characters, and emoji. The exported file was read back into pandas to confirm that its columns and row count matched the in-memory dataframe.

## 5. Cleaning results

| Year corpus | Raw total | Blank captions removed | Within-year duplicates removed | Total removed | Cleaned total |
|---:|---:|---:|---:|---:|---:|
| 2019 | 4,463 | 2 | 316 | 318 | 4,145 |
| 2023 | 3,578 | 1 | 98 | 99 | 3,479 |
| 2024 | 3,159 | 0 | 42 | 42 | 3,117 |
| 2025 | 3,391 | 0 | 66 | 66 | 3,325 |
| **Total** | **14,591** | **3** | **522** | **525** | **14,066** |

After the four cleaned corpora were combined, 17 further cross-year duplicates were removed. The final dataset contains **14,049 records**.

The final source-label distribution differs slightly from the per-year cleaned totals because cross-year duplicates retain the first occurrence according to the concatenation order:

| Source hashtag | Final records |
|---|---:|
| `klscm2019` | 4,145 |
| `klscm2023` | 3,476 |
| `klscm2024` | 3,111 |
| `klscm2025` | 3,317 |
| **Total** | **14,049** |

## 6. Validation checks

The notebook uses assertions and explicit summaries to verify that:

1. all required source files are present;
2. yearly counts reconcile as `raw total - total removed = cleaned total`;
3. the combined pre-deduplication count equals the sum of yearly cleaned counts;
4. the final columns occur in the required order;
5. no retained caption is null or blank;
6. no duplicate `author_id` and `caption` pairs remain;
7. only the four expected source hashtags are present;
8. every timestamp is parseable;
9. the exported CSV has the expected columns and 14,049 rows.

The completed CSV passed these checks and contains no missing values in the five retained fields.

## 7. Methodological limitations

- **Hashtag sampling:** The data represents posts discoverable through selected hashtags, not all Instagram discussion about KLSCM. Posts without these tags are absent.
- **Collection-time effects:** The files were collected in July 2026, so availability, account privacy, deletion, platform ranking, and scraper behavior may affect which historical posts were retrieved.
- **Event year ambiguity:** Source labels are based on retrieval hashtags. Publication timestamps extend beyond the named event years, and some captions mention multiple editions.
- **Duplicate definition:** Exact author-caption matching does not identify near-duplicates, lightly edited reposts, copied content across different authors, or the same sentiment expressed in different words.
- **Cross-year precedence:** When an author-caption pair occurs in several source files, the earliest file in the processing order retains the record and its source label.
- **Text preservation:** No language detection, translation, spelling correction, emoji conversion, URL removal, hashtag removal, case folding, or token-level preprocessing was performed. Those choices should be made later according to the requirements of the selected sentiment model.
- **Unit of analysis:** The final row represents a unique author-caption pair under the chosen rule, not necessarily a unique person, post URL, or independent opinion.

These limitations should be reported when interpreting differences in sentiment between source corpora.

## 8. Reproducibility and recommended journal description

The raw JSON files remain unchanged. All transformations are recorded in the notebook, and the cleaned output can be regenerated by running its cells sequentially. For reproducible reporting, the source filenames, collection date, pandas version, duplicate key, blank-caption rule, timestamp convention, and final record count should be retained in the research documentation.

A concise journal methodology statement may read:

> Instagram posts retrieved using the hashtags #klscm2019, #klscm2023, #klscm2024, and #klscm2025 were standardized using Python and pandas. Two source schemas were mapped to a common structure containing author ID, post URL, caption, source hashtag, and UTC timestamp. Captions containing no text were excluded, and leading and trailing whitespace was removed without otherwise modifying textual content. Duplicate observations were defined as records with the same author ID and caption and were removed first within each hashtag corpus and then across the combined dataset. From 14,591 raw records, 525 records were removed during corpus-level cleaning and 17 additional cross-corpus duplicates were removed, yielding 14,049 records for subsequent analysis. The hashtag-year labels denote retrieval corpora rather than strict publication-year samples.

## 9. Output artifacts

- `data_cleaning.ipynb`: executable cleaning and validation workflow.
- `instagram_cleanded.csv`: final analysis-ready dataset with 14,049 rows and five columns.
- `data_cleaning_report.md`: methodological record of the process, decisions, results, and limitations.
