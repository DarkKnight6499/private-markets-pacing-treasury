# Compliance Fail

Synthetic data and stylized assumptions.

| rule_id | kind | status | measured | lower | upper | warn_at | limit | basis |
| - | - | - | - | - | - | - | - | - |
| strategy_buyout | band | pass | 0.120000 | 0.000000 | 0.200000 | 0.180000 | [0.000000, 0.200000] | NAV plus unfunded divided by total portfolio NAV |
| strategy_secondaries | band | pass | 0.038000 | 0.000000 | 0.100000 | 0.090000 | [0.000000, 0.100000] | NAV plus unfunded divided by total portfolio NAV |
| strategy_co_invest | band | fail | 0.180000 | 0.000000 | 0.080000 | 0.072000 | [0.000000, 0.080000] | NAV plus unfunded divided by total portfolio NAV |
| vintage_current | band | fail | 0.265000 | 0.000000 | 0.200000 | 0.180000 | [0.000000, 0.200000] | NAV plus unfunded divided by total portfolio NAV |
| geography_na | band | pass | 0.403000 | 0.000000 | 0.450000 | 0.405000 | [0.000000, 0.450000] | NAV plus unfunded divided by total portfolio NAV |
| single_manager | concentration | fail | 0.280000 | 0.000000 | 0.180000 | 0.162000 | [0.000000, 0.180000] | NAV plus unfunded divided by total portfolio NAV |
| single_fund | concentration | fail | 0.160000 | 0.000000 | 0.150000 | 0.135000 | [0.000000, 0.150000] | NAV plus unfunded divided by total portfolio NAV |
| single_issuer | concentration | fail | 0.274900 | 0.000000 | 0.160000 | 0.144000 | [0.000000, 0.160000] | NAV plus unfunded divided by total portfolio NAV |
| side_letter_sector | exclusion | fail | 0.160000 | 0.000000 | 0.000000 | 0.000000 | [0.000000, 0.000000] | NAV plus unfunded divided by total portfolio NAV |
| side_letter_jurisdiction | exclusion | fail | 0.160000 | 0.000000 | 0.000000 | 0.000000 | [0.000000, 0.000000] | NAV plus unfunded divided by total portfolio NAV |
| unfunded_limit | unfunded | pass | 0.118000 | 0.000000 | 0.200000 | 0.180000 | [0.000000, 0.200000] | NAV plus unfunded divided by total portfolio NAV |
