---
description: "33 pre-built tools for file I/O, web requests, data processing, datetime, text, code, search, GitHub, and databases"
tags:
  - tools
  - built-in
---

# Toolbox: 33 Pre-Built Tools

**Import:** `from selectools.toolbox import get_all_tools`

**Stability:** stable

```python title="toolbox_quickstart.py"
from selectools import Agent, AgentConfig, Message, Role
from selectools.providers.stubs import LocalProvider
from selectools.toolbox import get_all_tools, get_tools_by_category

# Load all 33 pre-built tools across 9 categories
all_tools = get_all_tools()
print(f"Loaded {len(all_tools)} tools")

# Or load by category: file, web, data, datetime, text, code, search, github, db
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

**Added in:** v0.12.0 | **Expanded in:** v0.21.0

The toolbox provides **33 ready-to-use tools** across 9 categories that you can give to an agent immediately — no implementation needed.

---

## Quick Start

```python
from selectools import Agent, AgentConfig, OpenAIProvider
from selectools.toolbox import get_all_tools

agent = Agent(
    tools=get_all_tools(),           # all 33 tools
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

tools = get_all_tools()  # List[Tool], 33 tools
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
db_tools     = get_tools_by_category("db")         # 2 tools  (v0.21.0)
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
| `get_all_tools()` | Returns all 33 tools as `List[Tool]` |
| `get_tools_by_category(category)` | Returns tools for one category (`"file"`, `"web"`, `"data"`, `"datetime"`, `"text"`, `"code"`, `"search"`, `"github"`, `"db"`) |
| `selectools.toolbox.file_tools` | Module with 5 file tools |
| `selectools.toolbox.web_tools` | Module with 2 web tools |
| `selectools.toolbox.data_tools` | Module with 6 data tools |
| `selectools.toolbox.datetime_tools` | Module with 4 datetime tools |
| `selectools.toolbox.text_tools` | Module with 7 text tools |
| `selectools.toolbox.code_tools` | Module with 2 code execution tools (v0.21.0) |
| `selectools.toolbox.search_tools` | Module with 2 web search tools (v0.21.0) |
| `selectools.toolbox.github_tools` | Module with 3 GitHub tools (v0.21.0) |
| `selectools.toolbox.db_tools` | Module with 2 database tools (v0.21.0) |

---

## See Also

- [examples/03_toolbox.py](https://github.com/johnnichev/selectools/blob/main/examples/03_toolbox.py) — Working demo of all categories
- [TOOLS.md](TOOLS.md) — Creating your own tools with `@tool`
- [DYNAMIC_TOOLS.md](DYNAMIC_TOOLS.md) — Loading tools from files/directories at runtime

---

## Related Examples

| # | Script | Description |
|---|--------|-------------|
| 03 | [`03_toolbox.py`](https://github.com/johnnichev/selectools/blob/main/examples/03_toolbox.py) | Working demo of all 33 pre-built tools across 9 categories |
| 13 | [`13_dynamic_tools.py`](https://github.com/johnnichev/selectools/blob/main/examples/13_dynamic_tools.py) | Loading tools dynamically from files and directories |
