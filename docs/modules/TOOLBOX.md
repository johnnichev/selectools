---
description: "48 pre-built tools for file I/O, web requests, data processing, datetime, text, code, search, GitHub, databases, calculator, email, PDF, Slack, Notion, and Linear"
tags:
  - tools
  - built-in
---

# Toolbox: 48 Pre-Built Tools

**Import:** `from selectools.toolbox import get_all_tools`

**Stability:** stable

```python title="toolbox_quickstart.py"
from selectools import Agent, AgentConfig, Message, Role
from selectools.providers.stubs import LocalProvider
from selectools.toolbox import get_all_tools, get_tools_by_category

# Load all 48 pre-built tools across 15 categories
all_tools = get_all_tools()
print(f"Loaded {len(all_tools)} tools")

# Or load by category: file, web, data, datetime, text, code, search, github,
# database, calculator, email, pdf, slack, notion, linear
text_tools = get_tools_by_category("text")
data_tools = get_tools_by_category("data")

provider = LocalProvider()
agent = Agent(
    tools=text_tools + data_tools,
    provider=provider,
    config=AgentConfig(max_iterations=3),
)

result = agent.run([
    Message(role=Role.USER, content="Count the words in 'Hello world from selectools'")
])
print(result.content)
```

!!! tip "See Also"
    - [Tools](TOOLS.md) - Creating custom tools with the `@tool` decorator
    - [Dynamic Tools](DYNAMIC_TOOLS.md) - Loading tools from files and directories at runtime

---

**Added in:** v0.12.0 | **Expanded in:** v0.21.0, v0.23.0

The toolbox provides **48 ready-to-use tools** across 15 categories that you can give to an agent immediately — no implementation needed.

---

## Quick Start

```python
from selectools import Agent, AgentConfig, OpenAIProvider
from selectools.toolbox import get_all_tools

agent = Agent(
    tools=get_all_tools(),           # all 48 tools
    provider=OpenAIProvider(),
    config=AgentConfig(max_iterations=5),
)

result = agent.ask("Read the file config.json and extract the 'database.host' field")
print(result.content)
```

---

## Loading Tools

### All Tools

```python
from selectools.toolbox import get_all_tools

tools = get_all_tools()  # List[Tool], 48 tools
```

### By Category

```python
from selectools.toolbox import get_tools_by_category

file_tools   = get_tools_by_category("file")       # 5 tools
web_tools    = get_tools_by_category("web")        # 2 tools
data_tools   = get_tools_by_category("data")       # 6 tools
dt_tools     = get_tools_by_category("datetime")   # 4 tools
text_tools   = get_tools_by_category("text")       # 7 tools
code_tools   = get_tools_by_category("code")       # 2 tools  (v0.21.0)
search_tools = get_tools_by_category("search")     # 2 tools  (v0.21.0)
gh_tools     = get_tools_by_category("github")     # 3 tools  (v0.21.0)
db_tools     = get_tools_by_category("database")   # 2 tools  (v0.21.0)
calc_tools   = get_tools_by_category("calculator") # 2 tools  (v0.23.0)
email_tools  = get_tools_by_category("email")      # 2 tools  (v0.23.0)
pdf_tools    = get_tools_by_category("pdf")        # 2 tools  (v0.23.0)
slack_tools  = get_tools_by_category("slack")      # 3 tools  (v0.23.0)
notion_tools = get_tools_by_category("notion")     # 3 tools  (v0.23.0)
linear_tools = get_tools_by_category("linear")     # 3 tools  (v0.23.0)
```

### Individual Tools

```python
from selectools.toolbox.file_tools import read_file, write_file
from selectools.toolbox.web_tools import http_get
from selectools.toolbox.data_tools import parse_json, json_to_csv
from selectools.toolbox.text_tools import extract_emails, convert_case
from selectools.toolbox.datetime_tools import get_current_time
from selectools.toolbox.code_tools import execute_python, execute_shell       # v0.21.0
from selectools.toolbox.search_tools import web_search, scrape_url            # v0.21.0
from selectools.toolbox.github_tools import github_search_repos, github_get_file  # v0.21.0
from selectools.toolbox.db_tools import query_sqlite, query_postgres          # v0.21.0
from selectools.toolbox.calculator_tools import evaluate_expression, unit_convert  # v0.23.0
from selectools.toolbox.email_tools import send_email, read_inbox             # v0.23.0
from selectools.toolbox.pdf_tools import extract_pdf_text, extract_pdf_tables  # v0.23.0
from selectools.toolbox.slack_tools import slack_send_message, slack_read_channel  # v0.23.0
from selectools.toolbox.notion_tools import notion_create_page, notion_search  # v0.23.0
from selectools.toolbox.linear_tools import linear_create_issue, linear_list_issues  # v0.23.0
```

---

## File Tools (5)

| Tool | Description | Parameters |
|---|---|---|
| `read_file` | Read a text file | `filepath`, `encoding="utf-8"` |
| `write_file` | Write/append text to a file | `filepath`, `content`, `mode="w"`, `encoding` |
| `list_files` | List files matching a glob pattern | `directory="."`, `pattern="*"`, `show_hidden=False`, `recursive=False` |
| `file_exists` | Check if a path exists | `path` |
| `read_file_stream` | Stream file line-by-line (streaming tool) | `filepath`, `encoding` |

```python
from selectools.toolbox import get_tools_by_category

agent = Agent(
    tools=get_tools_by_category("file"),
    provider=provider,
    config=AgentConfig(max_iterations=5),
)

agent.ask("Write 'Hello World' to output.txt, then read it back")
agent.ask("List all .py files in the src/ directory recursively")
```

`read_file_stream` is a **streaming tool** — it yields lines progressively, which is useful for large files. See [STREAMING.md](STREAMING.md) for more on streaming tools.

---

## Web Tools (2)

| Tool | Description | Parameters |
|---|---|---|
| `http_get` | HTTP GET request | `url`, `headers=None` (JSON string), `timeout=30` |
| `http_post` | HTTP POST with JSON body | `url`, `data` (JSON string), `headers=None`, `timeout=30` |

Requires the `requests` library (`pip install requests`).

```python
agent = Agent(
    tools=get_tools_by_category("web"),
    provider=provider,
    config=AgentConfig(max_iterations=3),
)

agent.ask("Fetch https://api.github.com/repos/python/cpython")
agent.ask("POST to https://httpbin.org/post with data {\"name\": \"test\"}")
```

JSON responses are automatically pretty-printed. Long text responses are truncated to 5000 characters.

---

## Data Tools (6)

| Tool | Description | Parameters |
|---|---|---|
| `parse_json` | Validate and pretty-print JSON | `json_string`, `pretty=True` |
| `json_to_csv` | Convert JSON array to CSV | `json_string`, `delimiter=","` |
| `csv_to_json` | Convert CSV to JSON array | `csv_string`, `delimiter=","`, `pretty=True` |
| `extract_json_field` | Extract field by dot-path | `json_string`, `field_path` (e.g. `"user.name"`, `"items.0.price"`) |
| `format_table` | Render JSON array as table | `data` (JSON string), `format_type="simple"` / `"markdown"` / `"csv"` |
| `process_csv_stream` | Stream CSV rows (streaming tool) | `filepath`, `delimiter=","`, `encoding` |

```python
agent = Agent(
    tools=get_tools_by_category("data"),
    provider=provider,
    config=AgentConfig(max_iterations=5),
)

agent.ask('Parse this JSON and convert to CSV: [{"name":"Alice","age":30},{"name":"Bob","age":25}]')
agent.ask('Extract the "items.0.price" field from {"items":[{"price":9.99}]}')
```

`process_csv_stream` is a **streaming tool** for large CSV files.

---

## DateTime Tools (4)

| Tool | Description | Parameters |
|---|---|---|
| `get_current_time` | Current date/time | `timezone="UTC"`, `format="%Y-%m-%d %H:%M:%S %Z"` |
| `parse_datetime` | Parse a date string | `datetime_string`, `input_format=None`, `output_format` |
| `time_difference` | Diff between two dates | `start_date`, `end_date`, `unit="days"` / `"hours"` / `"minutes"` / `"seconds"` |
| `date_arithmetic` | Add/subtract from a date | `date`, `operation="add"` / `"subtract"`, `value`, `unit="days"` |

Timezone support requires `pytz` (`pip install pytz`). UTC works without it.

```python
agent = Agent(
    tools=get_tools_by_category("datetime"),
    provider=provider,
    config=AgentConfig(max_iterations=3),
)

agent.ask("What's the current time in America/New_York?")
agent.ask("How many days between 2026-01-01 and 2026-12-31?")
agent.ask("What date is 90 days from 2026-03-12?")
```

`parse_datetime` automatically tries 12 common date formats when `input_format` is not specified.

---

## Text Tools (7)

| Tool | Description | Parameters |
|---|---|---|
| `count_text` | Count words, characters, lines | `text`, `detailed=True` |
| `search_text` | Regex search | `text`, `pattern`, `case_sensitive=True`, `return_matches=True` |
| `replace_text` | Regex replace | `text`, `pattern`, `replacement`, `case_sensitive=True`, `max_replacements=0` |
| `extract_emails` | Find email addresses | `text` |
| `extract_urls` | Find URLs | `text` |
| `convert_case` | Change case | `text`, `case_type` (`upper`, `lower`, `title`, `sentence`, `camel`, `snake`, `kebab`) |
| `truncate_text` | Truncate with suffix | `text`, `max_length=100`, `suffix="..."` |

```python
agent = Agent(
    tools=get_tools_by_category("text"),
    provider=provider,
    config=AgentConfig(max_iterations=3),
)

agent.ask("Extract all emails and URLs from: 'Contact support@example.com at https://example.com'")
agent.ask("Convert 'hello world example' to camelCase")
agent.ask("Count the words in this paragraph: ...")
```

---

## Code Tools (2) — v0.21.0

| Tool | Description | Parameters |
|---|---|---|
| `execute_python` | Execute Python code in a subprocess | `code`, `timeout=30` |
| `execute_shell` | Execute a shell command | `command`, `timeout=30` |

!!! warning "Security"
    Code execution tools run commands on the host machine. Use `ToolPolicy` to restrict access or require human approval:

    ```python
    from selectools.policy import ToolPolicy

    policy = ToolPolicy(review=["execute_*"])  # Require approval before execution
    ```

```python
agent = Agent(
    tools=get_tools_by_category("code"),
    provider=provider,
    config=AgentConfig(max_iterations=3),
)

agent.ask("Run this Python code: print('Hello from subprocess!')")
```

Output is truncated to 10 KB. Maximum timeout is 300 seconds.

---

## Search Tools (2) — v0.21.0

| Tool | Description | Parameters |
|---|---|---|
| `web_search` | Search the web via DuckDuckGo (no API key) | `query`, `num_results=5` |
| `scrape_url` | Fetch a URL and extract text content | `url`, `selector=None` |

No external dependencies required -- uses `urllib` from the standard library.

```python
agent = Agent(
    tools=get_tools_by_category("search"),
    provider=provider,
    config=AgentConfig(max_iterations=3),
)

agent.ask("Search for 'Python async programming best practices'")
agent.ask("Scrape the text from https://example.com")
```

---

## GitHub Tools (3) — v0.21.0

| Tool | Description | Parameters |
|---|---|---|
| `github_search_repos` | Search GitHub repositories | `query`, `max_results=5` |
| `github_get_file` | Get file contents from a repository | `repo`, `path`, `ref="main"` |
| `github_list_issues` | List issues in a repository | `repo`, `state="open"`, `max_results=10` |

Uses the GitHub REST API v3. Set the `GITHUB_TOKEN` environment variable for authenticated requests (higher rate limits).

```python
agent = Agent(
    tools=get_tools_by_category("github"),
    provider=provider,
    config=AgentConfig(max_iterations=3),
)

agent.ask("Search GitHub for 'machine learning language:python'")
agent.ask("Get the README from johnnichev/selectools")
agent.ask("List open issues in johnnichev/selectools")
```

---

## Database Tools (2) — v0.21.0

| Tool | Description | Parameters |
|---|---|---|
| `query_sqlite` | Execute a read-only SQL query against SQLite | `db_path`, `sql`, `max_rows=100` |
| `query_postgres` | Execute a read-only SQL query against PostgreSQL | `connection_string`, `sql`, `max_rows=100` |

Both tools enforce **read-only mode** to prevent accidental writes. SQLite uses the standard-library `sqlite3` module. PostgreSQL requires `psycopg2` (`pip install psycopg2-binary`).

```python
agent = Agent(
    tools=get_tools_by_category("db"),
    provider=provider,
    config=AgentConfig(max_iterations=3),
)

agent.ask("Query the database at ./app.db: SELECT name, email FROM users LIMIT 5")
```

---

## Calculator Tools (2) — v0.23.0

| Tool | Description | Parameters |
|---|---|---|
| `evaluate_expression` | Safely evaluate a math expression | `expression` |
| `unit_convert` | Convert length/mass/temperature/data units | `value`, `from_unit`, `to_unit` |

No dependencies — pure standard library.

!!! info "Security"
    `evaluate_expression` parses input with `ast` and walks the tree against an
    explicit whitelist of nodes, operators, math functions, and constants. It never
    calls `eval()`/`exec()`, rejects names, attribute access, subscripts, lambdas,
    and comprehensions, and guards `**` operand sizes against memory/CPU bombs.

Supported: `+ - * / // % **`, parentheses, `pi`/`e`/`tau`, and `abs`, `round`,
`min`, `max`, `sqrt`, `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `log`, `log2`,
`log10`, `exp`, `floor`, `ceil`, `degrees`, `radians`.

```python
agent = Agent(
    tools=get_tools_by_category("calculator"),
    provider=provider,
    config=AgentConfig(max_iterations=3),
)

agent.ask("What is sqrt(2) * 10 ** 3?")
agent.ask("Convert 26.2 miles to km and 98.6 F to C")
```

Units: length (mm, cm, m, km, in, ft, yd, mi), mass (mg, g, kg, t, oz, lb),
temperature (c, f, k), data (b, kb, mb, gb, tb, kib, mib, gib, tib).

---

## Email Tools (2) — v0.23.0

| Tool | Description | Parameters |
|---|---|---|
| `send_email` | Send mail via SMTP with STARTTLS (or SSL on port 465) | `to`, `subject`, `body`, `smtp_host`, `smtp_port`, `username`, `password`, `from_addr` |
| `read_inbox` | Read latest messages via IMAP (read-only) | `limit=5`, `imap_host`, `imap_port`, `username`, `password`, `folder="INBOX"` |

Standard library only (`smtplib`/`imaplib`). Connection settings fall back to
environment variables: `SMTP_HOST`, `SMTP_PORT`, `SMTP_USERNAME`, `SMTP_PASSWORD`,
`SMTP_FROM` and `IMAP_HOST`, `IMAP_PORT`, `IMAP_USERNAME`, `IMAP_PASSWORD`.

!!! warning "Credentials"
    Prefer env vars over passing passwords as tool parameters — parameters flow
    through the LLM conversation. Passwords are never echoed in tool output or
    error messages.

```python
agent = Agent(
    tools=get_tools_by_category("email"),
    provider=provider,
    config=AgentConfig(max_iterations=3),
)

agent.ask("Email a status summary to team@example.com")
agent.ask("Summarize my 5 most recent emails")
```

---

## PDF Tools (2) — v0.23.0

| Tool | Description | Parameters |
|---|---|---|
| `extract_pdf_text` | Extract text from a PDF | `path`, `pages=None` (e.g. `"1-3,5"`) |
| `extract_pdf_tables` | Extract tables as pipe-delimited rows | `path`, `pages=None` |

Requires `pdfplumber` (`pip install selectools[toolbox]`). The import is lazy —
the toolbox loads fine without it.

```python
agent = Agent(
    tools=get_tools_by_category("pdf"),
    provider=provider,
    config=AgentConfig(max_iterations=3),
)

agent.ask("Extract the tables from invoice.pdf pages 1-2 and total the amounts")
```

---

## Slack Tools (3) — v0.23.0

| Tool | Description | Parameters |
|---|---|---|
| `slack_send_message` | Send a message to a channel or DM | `channel`, `text`, `token=None` |
| `slack_read_channel` | Read recent channel history | `channel`, `limit=10`, `token=None` |
| `slack_search_messages` | Search messages across the workspace | `query`, `count=10`, `token=None` |

Requires `slack-sdk` (`pip install selectools[toolbox]`). Token via the
`SLACK_BOT_TOKEN` env var or the `token` parameter. `slack_search_messages`
needs a *user* token (`xoxp-`) with the `search:read` scope — Slack does not
allow bot tokens on the search API.

```python
agent = Agent(
    tools=get_tools_by_category("slack"),
    provider=provider,
    config=AgentConfig(max_iterations=3),
)

agent.ask("Post 'deploy finished' to #releases")
agent.ask("Summarize the last 20 messages in C0123456789")
```

---

## Notion Tools (3) — v0.23.0

| Tool | Description | Parameters |
|---|---|---|
| `notion_create_page` | Create a page under a parent page | `parent_page_id`, `title`, `content=""`, `api_key=None` |
| `notion_search` | Search pages/databases shared with the integration | `query`, `max_results=5`, `api_key=None` |
| `notion_update_page` | Rename and/or archive a page | `page_id`, `title=None`, `archived=None`, `api_key=None` |

Uses the Notion REST API v1 via `requests` (`pip install selectools[toolbox]`).
Token via the `NOTION_API_KEY` env var. The integration must be shared with the
pages it operates on.

```python
agent = Agent(
    tools=get_tools_by_category("notion"),
    provider=provider,
    config=AgentConfig(max_iterations=3),
)

agent.ask("Find my 'Roadmap' page and add a child page titled 'Q3 Plan'")
```

---

## Linear Tools (3) — v0.23.0

| Tool | Description | Parameters |
|---|---|---|
| `linear_create_issue` | Create an issue in a team | `team_id`, `title`, `description=""`, `api_key=None` |
| `linear_list_issues` | List recent issues (optionally by team) | `team_id=None`, `limit=10`, `api_key=None` |
| `linear_update_issue` | Update title/description/state | `issue_id`, `title=None`, `description=None`, `state_id=None`, `api_key=None` |

Uses the Linear GraphQL API via `requests` (`pip install selectools[toolbox]`).
Key via the `LINEAR_API_KEY` env var.

```python
agent = Agent(
    tools=get_tools_by_category("linear"),
    provider=provider,
    config=AgentConfig(max_iterations=3),
)

agent.ask("List my open issues and create a follow-up issue for the flaky test")
```

---

## Combining with Custom Tools

Toolbox tools are regular `Tool` objects — mix them freely with your own:

```python
from selectools import tool
from selectools.toolbox import get_tools_by_category

@tool(description="Query our internal database")
def query_db(sql: str) -> str:
    # your custom implementation
    return "results..."

agent = Agent(
    tools=[query_db] + get_tools_by_category("data") + get_tools_by_category("text"),
    provider=provider,
    config=AgentConfig(max_iterations=5),
)
```

---

## API Reference

| Function | Description |
|---|---|
| `get_all_tools()` | Returns all 48 tools as `List[Tool]` |
| `get_tools_by_category(category)` | Returns tools for one category (`"file"`, `"web"`, `"data"`, `"datetime"`, `"text"`, `"code"`, `"search"`, `"github"`, `"database"`, `"calculator"`, `"email"`, `"pdf"`, `"slack"`, `"notion"`, `"linear"`) |
| `selectools.toolbox.file_tools` | Module with 5 file tools |
| `selectools.toolbox.web_tools` | Module with 2 web tools |
| `selectools.toolbox.data_tools` | Module with 6 data tools |
| `selectools.toolbox.datetime_tools` | Module with 4 datetime tools |
| `selectools.toolbox.text_tools` | Module with 7 text tools |
| `selectools.toolbox.code_tools` | Module with 2 code execution tools (v0.21.0) |
| `selectools.toolbox.search_tools` | Module with 2 web search tools (v0.21.0) |
| `selectools.toolbox.github_tools` | Module with 3 GitHub tools (v0.21.0) |
| `selectools.toolbox.db_tools` | Module with 2 database tools (v0.21.0) |
| `selectools.toolbox.calculator_tools` | Module with 2 calculator tools (v0.23.0) |
| `selectools.toolbox.email_tools` | Module with 2 email tools (v0.23.0) |
| `selectools.toolbox.pdf_tools` | Module with 2 PDF tools (v0.23.0) |
| `selectools.toolbox.slack_tools` | Module with 3 Slack tools (v0.23.0) |
| `selectools.toolbox.notion_tools` | Module with 3 Notion tools (v0.23.0) |
| `selectools.toolbox.linear_tools` | Module with 3 Linear tools (v0.23.0) |

---

## See Also

- [examples/03_toolbox.py](https://github.com/johnnichev/selectools/blob/main/examples/03_toolbox.py) — Working demo of all categories
- [TOOLS.md](TOOLS.md) — Creating your own tools with `@tool`
- [DYNAMIC_TOOLS.md](DYNAMIC_TOOLS.md) — Loading tools from files/directories at runtime

---

## Related Examples

| # | Script | Description |
|---|--------|-------------|
| 03 | [`03_toolbox.py`](https://github.com/johnnichev/selectools/blob/main/examples/03_toolbox.py) | Working demo of all 48 pre-built tools across 15 categories |
| 13 | [`13_dynamic_tools.py`](https://github.com/johnnichev/selectools/blob/main/examples/13_dynamic_tools.py) | Loading tools dynamically from files and directories |
| 104 | [`104_toolbox_expansion.py`](https://github.com/johnnichev/selectools/blob/main/examples/104_toolbox_expansion.py) | Calculator, email, PDF, Slack, Notion, and Linear tools (v0.23.0) |
