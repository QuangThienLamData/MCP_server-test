# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
Đây đồng thời là **bản thiết kế (design spec)** cho Competitor Research MCP Server.

## Mục tiêu

Một FastAPI app host nhiều MCP server độc lập (mounted sub-app), dùng chung Pinecone làm vector store, phục vụ team Growth & Marketing nghiên cứu đối thủ: chiến lược, app-store review, UX pattern, và xu hướng ngành. Deploy trên Render (Docker, port 10000), Python 3.12, quản lý bằng `uv`.

> **Các quyết định thiết kế đã chốt (đọc trước khi build):**
> 1. **Crawl trang JS bằng Jina Reader** (`https://r.jina.ai/`), **KHÔNG dùng Playwright** — Render giới hạn 512MB, Chromium sẽ OOM (xem commit `2bf7837`). Pattern này đã có trong `rag_mcp.py`.
> 2. **Không có scheduler chạy in-process.** Mỗi subsystem index expose tool `trigger_crawl`; lịch (7am / 7am+7pm) do **Render Cron Job (service riêng)** hoặc cron ngoài gọi vào tool này. Render web service ngủ khi không có traffic nên `schedule`/APScheduler in-process sẽ không nổ.
> 3. **Giữ nguyên lớp Auth hiện tại**: FastAPI + Scalekit OAuth (`src/auth.py`), transport streamable-HTTP, scope `search:read` cho `tools/call`. Mỗi subsystem mount thêm phải có route `.well-known/...` + nhánh trong `auth.py`.
> 4. **Xử lý đa ngôn ngữ** (yêu cầu: data nguồn bất kỳ ngôn ngữ nào, câu trả lời phải là Việt/Anh) — xem mục [Đa ngôn ngữ](#đa-ngôn-ngữ).
> 5. **Pinecone `us-east-1`**, 1 index nhiều namespace (free/starter plan chỉ hỗ trợ us-east-1).
> 6. Scraper store (Google Play / App Store) chấp nhận giới hạn + có retry (IP datacenter dễ bị chặn, iOS RSS chỉ ~500 review gần nhất).
> 7. Refero: server đóng vai **MCP client** gọi ra Refero MCP *bên trong* tool (không phải config client-side).

## Commands

```bash
uv sync                      # cài dependencies từ uv.lock
uv run main.py               # chạy toàn bộ app (tất cả MCP server) trên port 10000
uv run rag_mcp.py            # chạy 1 server standalone qua stdio (có __main__ block) để test cục bộ

# Docker (khớp deploy Render):
docker build -t mcp-server . && docker run -p 10000:10000 --env-file .env mcp-server
```

Chưa có test suite / linter / formatter.

## Kiến trúc tổng quan

```
                    ┌─────────────────────────────────────────┐
                    │  FastAPI host (main.py) + AuthMiddleware  │
                    │  Scalekit OAuth · scope search:read       │
                    └───┬─────────┬─────────┬─────────┬─────────┘
              mount →   │/strategy│ /reviews│  /ux    │ /news    (mỗi cái = 1 FastMCP)
                    [Sub A]   [Sub B]   [Sub C]   [Sub D]
                  Strategy   AppReview  Refero   News/Trends
                     RAG       RAG      (MCP     (GNews)
                        │        │      client)     │
                        └────────┴─────────┬────────┘
                            ┌──────────────────────────────┐
                            │  Pinecone (1 index, us-east-1)│
                            │  namespaces:                  │
                            │  strategy | app_reviews       │
                            │  ux_patterns | news_trends    │
                            └──────────────────────────────┘
                            ┌──────────────────────────────┐
                            │  External: OpenAI embed ·     │
                            │  Tavily · GNews · Refero ·    │
                            │  Jina Reader (JS pages)       │
                            └──────────────────────────────┘
```

**Pattern host + mounted sub-app** (đã có trong repo): mỗi MCP server là 1 module export `mcp = FastMCP(...)` với các `@mcp.tool()`. `main.py` mount `streamable_http_app()` của từng server dưới path prefix và chạy session manager qua `lifespan`. Hiện có: `gnews_mcp_server.py` (`/gnews`), `email_mcp.py` (`/email`, placeholder), `rag_mcp.py` (`/rag`).

**Thêm 1 MCP server mới** (bắt buộc đủ 3 bước để qua được Auth):
1. Tạo `foo_mcp.py`: `mcp = FastMCP(name=..., transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False))` + các `@mcp.tool()`.
2. `main.py`: import, thêm dict `FOO_METADATA`, route `/.well-known/oauth-protected-resource/foo/mcp`, `enter_async_context(foo.session_manager.run())` trong `lifespan`, `app.mount("/foo", ...)`.
3. `src/auth.py`: thêm nhánh `metadata_url` cho path `/foo`.

## Hạ tầng dùng chung (viết trước)

- **Embedding**: OpenAI `text-embedding-3-small` (1536 dims). Có sẵn `_embed()` trong `rag_mcp.py`.
- **Pinecone**: 1 index cosine serverless `us-east-1`, tách dữ liệu bằng `namespace` (không tạo nhiều index). Có sẵn `_get_index()` (lazy-init, auto-create).
- **ID vector**: **luôn** dùng `hashlib.md5(x.encode()).hexdigest()` — KHÔNG dùng builtin `hash()` (trả int không slice được + bị salt ngẫu nhiên mỗi process).
- **Dedup**: content-hash MD5 lưu trong SQLite `crawled_pages` (đã có trong `rag_mcp.py`), skip trang không đổi.
- **Fetch trang JS**: `_fetch_rendered()` qua Jina Reader (đã có). Trang tĩnh: `httpx` + BeautifulSoup.
- **Tavily**: dùng cho web-search fallback (`search`) và lấy full content (`extract`).

### Đa ngôn ngữ

Yêu cầu: nguồn có thể bất kỳ ngôn ngữ nào, nhưng Claude phải trả lời **tiếng Việt hoặc tiếng Anh**. Hai vế:
- **Câu trả lời**: MCP server chỉ trả *dữ liệu thô*; ngôn ngữ trả lời do Claude quyết định. Để ép đúng, thêm câu chỉ dẫn vào **docstring của mỗi tool** / server instruction, ví dụ: *"Answer in Vietnamese or English regardless of the source language of the content."* Server **không** tự dịch.
- **Retrieval (quan trọng)**: `text-embedding-3-small` làm cross-lingual **yếu** (hỏi tiếng Việt, doc tiếng Anh → dễ miss). Bắt buộc:
  - Khi index: lưu thêm field `summary` **tiếng Việt** (~100 từ) trong metadata để hiển thị/đối chiếu nhanh.
  - Khi query: cân nhắc dịch/chuẩn hoá query trước khi embed để tăng recall.

## Sub A — Strategy Research RAG · `rag_mcp.py` · mount `/rag` · namespace mặc định

Crawl nội dung đối thủ (`competitors.json`: ZaloPay/ShopeePay/VNPay…) → chunk → embed → Pinecone + dedup SQLite. **Đã có + đã bổ sung cross-lingual & Tavily fallback.**

Tools thật: `search_competitor_content`, `web_search_and_index`, `list_competitor_topics`, `compare_competitor_messaging`, `get_crawl_status`, `trigger_crawl`. Metadata vector: `competitor_name`, `url`, `title`, `source_type` (blog/rss/web_search…), `published_at`, `crawled_at`, `chunk_index`, `text`.

**Đã implement + test (turn này):**
- **Cross-lingual**: `search_competitor_content` dùng `_bilingual_queries` (đã chuyển xuống `rag_mcp.py` làm hàm chung, `news_mcp` import lại — tránh circular) → embed EN+VI → merge score max. Test thật: VI "khuyến mãi ShopeePay" (score 0.72) và **EN "ZaloPay promotions" chạm đúng nội dung tiếng Việt** (0.70).
- **Tavily fallback** (đã test LIVE): gọi REST bằng `httpx` (không thêm dep) — `_tavily_search` (POST `api.tavily.com/search`, `api_key` trong body; **lọc `score >= TAVILY_MIN_SCORE` 0.5** vì Tavily hay xếp bài lạc đề cao ở query mơ hồ) → `_index_tavily_results` (id = md5(url)+"_web", upsert namespace mặc định, tự enrich) → `_format_web_results`. `search_competitor_content` tự gọi khi best score < `FALLBACK_SCORE_THRESHOLD` (0.35, KHÔNG phải 0.75 như draft — score thực tế ~0.4–0.7) và có key; tool `web_search_and_index` để gọi thủ công.
- `TAVILY_API_KEY` **đã có trong `.env`**. Không có key thì fallback tự bỏ qua (degrade an toàn). Test live: `competitor_name="MoMo"` (0 bài trong DB) → fallback → trả khuyến mãi MoMo từ momo.vn; `web_search_and_index("MoMo super app funding")` → MoMo Series D. Vector test đã dọn, prod giữ nguyên 470.
- Lưu ý: dữ liệu đối thủ (~419 vector) đang có sẵn trên Pinecone namespace mặc định từ crawl trước; bảng SQLite `crawled_pages` chỉ tạo khi `trigger_crawl`/startup chạy (local hiện chưa có bảng này).

## Sub B — App Store Rating RAG · `reviews_mcp.py` · mount `/reviews` · namespace `app_reviews`

Crawl review Google Play + App Store → Pinecone (namespace `app_reviews`) + SQLite (bảng `app_reviews`). **Đã implement + test.** Giữ nguyên code crawl của user (`gp_get_reviews`, `ios_get_reviews`, `ios_get_reviews_rss`, `run_google_play`, `run_appstore`) — **đã bỏ hết BigQuery + pandas**. Dep mới: `google-play-scraper`.

Apps trong `apps.json` (android + iOS, đã verify hợp lệ trên kho VN):
- MoMo — android `com.mservice.momotransfer`, ios `918751511`
- ZaloPay — android `vn.com.vng.zalopay`, ios `1112407590`
- VNPay — android `vnpay.smartacccount`, ios `1470378562`
- ShopeePay — android `com.beeasy.toppay`, ios `1032301823`

Lưu ý: android id của ZaloPay/VNPay trong draft ban đầu (`com.vng.zalopay`, `vn.com.vnpay.customers`) **sai/404** — đã sửa. Dùng iTunes Search API (search theo tên, country=vn, entity=software) và `google_play_scraper.search`/`app` để tra; chọn app người dùng (không phải Merchant/CA/Authenticator).

ID vector = `{platform}_{app}_{reviewId}`. Metadata: `app`, `platform`, `rating`, `review_date`, `version`, `language` (detect từ text), `has_reply`, `title`, **`text`** (CÓ lưu để search hiển thị — review ngắn, dưới xa giới hạn Pinecone; khác draft). SQLite giữ thêm `thumbs_up`, `user_name` cho aggregation.

Tools: `search_app_reviews` (bilingual + filter app/platform/rating_max/date), `get_review_insights` (LLM gpt-4o-mini tổng hợp top vấn đề + đề xuất, tiếng Việt), `compare_rating_trend` (phân bố sao + avg từ SQLite), `trigger_crawl(app?, platform?, last_days=3)` (background, cron gọi), `get_crawl_status`.

**Đã học khi test:**
- **iOS kho VN: RSS trả RỖNG** (`.../rss/customerreviews/...` storefront vn) → đã thêm **fallback sang amp-api** (`apps.apple.com/api/.../reviews`, header `authorization: Bearer` rỗng) khi RSS 0 kết quả, giới hạn bằng `max_scan` (mặc định 500) để không quét vô tận. RSS vẫn OK cho storefront khác (US trả 50).
- **amp-api `mostRecent` KHÔNG sắp xếp chặt theo ngày** → KHÔNG được early-break theo `from_date` (sẽ rớt review); chỉ bound bằng `max_scan`. Review iOS VN của MoMo rất thưa (vài bài/vài tháng).
- GP chạy tốt (MoMo 4.17★, 688k rating); review rỗng/quá ngắn bị lọc (`len<10`). Semantic + cross-lingual search OK (query "nạp tiền lỗi giao dịch thất bại" → đúng review 1★, score 0.67). `get_review_insights` cho bản tổng hợp tiếng Việt chuẩn (top themes + đề xuất).
- Có ~17 review MoMo (14 android + ~3 ios) đang nằm trong namespace `app_reviews` từ test; chạy `trigger_crawl` để lấy đầy đủ (dedup theo id).

## Sub C — Refero UX Integration · `refero_mcp.py` · mount `/ux`

Proxy tới Refero (135k+ màn hình app thật). Server đóng vai **MCP client**: `_refero_call` mở `streamablehttp_client` tới `REFERO_MCP_URL` (`https://api.refero.design/mcp`) với header `Authorization: Bearer <REFERO_TOKEN>`, initialize → call tool → trả text. **Đã implement + test.**

- **Auth**: chỉ cần Bearer token (đã verify: initialize + list_tools chạy headless). Browser OAuth mà Refero nhắc chỉ là flow của Claude Code client — **API không cần** cho server-to-server.
- `REFERO_MCP_URL` + `REFERO_TOKEN` trong `.env`.
- **`platform` chỉ nhận `ios` | `web`** (enum của Refero); `_platform()` coerce giá trị khác về `web`.
- Query auto-dịch sang tiếng Anh (`_english_query` dùng `_bilingual_queries`) vì Refero là DB tiếng Anh.
- Tools (thin proxy): `search_ux_patterns` (→refero_search_screens), `search_user_flows` (→refero_search_flows), `search_design_styles`, `get_ux_screen`, `get_ux_flow`. Tool async (`async def`).
- Xử lý lỗi `NO_SUBSCRIPTION` → trả thông báo activate tại refero.design/mcp/upgrade.
- **CHẶN hiện tại**: subscription Refero đang inactive/hết hạn → mọi call dữ liệu trả NO_SUBSCRIPTION. Auth + proxy + dịch query đã chạy đúng; chỉ chờ kích hoạt gói là ra data. Refero tools khác có sẵn: `refero_get_similar_screens`, `refero_get_screen_image`, `refero_get_style` (thêm wrapper khi cần).
- Caching kết quả vào namespace `ux_patterns` (giảm gọi Refero — Pro giới hạn ~8k calls/tháng): CHƯA làm (v1).

## Sub D — Industry News & Trends (GNews) · namespace `news_trends` · mount `/news`

Theo dõi fintech/AI/product/growth từ GNews. **Đã implement + test end-to-end trong `news_mcp.py`** (mount `/news`), tái dùng `_embed`/`_get_index`/`DB_PATH` từ `rag_mcp.py`. Topic/keyword cấu hình trong `news_topics.json` (JSON cho đồng bộ với `competitors.json`, không thêm dep PyYAML). Semantic search dùng Pinecone namespace `news_trends`; listing/digest dùng bảng SQLite `news_articles` (dedup theo `article_url`). GNews free plan cắt `content` ~200 ký tự → embed `title + description + content`; cần full thì `tavily.extract(url)`. Lưu ý: `/gnews` cũ vẫn còn và trùng chức năng realtime với `/news` — cân nhắc gỡ sau.

**Đã học khi test (giữ đúng, đừng lặp lại lỗi):**
- Keyword GNews: cụm nhiều từ trong query OR **phải bọc ngoặc kép** (`"e-wallet Vietnam" OR ZaloPay`), nếu không GNews trả **400**. Tránh term trần mơ hồ (`MoMo` → trúng bánh momo; `retention`/`gamification` → trúng horoscope/HR). Curation keyword là việc theo domain.
- Crawl **không lọc `from`**: GNews free coverage thưa, cửa sổ 24h/7d thường = 0; đã dựa vào newest-first + dedup URL.
- **Không index top-headlines theo category vào topic** (tin chung chung → làm bẩn relevance); top-headlines chỉ phục vụ tool realtime `get_top_headlines`.
- `search_industry_trends` (semantic) rank tốt kể cả corpus nhiễu (score ~0.6 cho bài đúng); `get_weekly_trend_digest` chỉ liệt kê theo thời gian nên phơi nhiễu — dùng semantic làm retrieval chính.
- **Language field detect từ text** (`_detect_lang`, regex ký tự riêng của tiếng Việt), KHÔNG tin `lang` của query — vì GNews free **không lọc `lang=vi`** (trả bài EN/ES/PT gán nhầm nhãn).

**Cross-lingual (đã implement + verify) — chạy 2 chiều:**
- `_bilingual_queries` dịch query sang EN+VI (gpt-4o-mini, JSON mode, có fallback), `search_industry_trends` embed cả 2 biến thể → query Pinecone → merge theo score max. Query VI kéo được doc EN và ngược lại (đã test: query EN "cashless payment in Vietnam" trả về đúng bài tiếng Việt).
- `news_topics.json` schema `queries: [{q, lang, country?}]` (EN + VI).
- `language` metadata **detect từ text** (`_detect_lang`), không tin nhãn query.

**Nguồn nội dung tiếng Việt = RSS báo VN (không phải GNews):**
- GNews free ~0 coverage VN → đã thêm crawl **RSS** trong `_fetch_feed_articles`: mỗi topic có `feeds: [{url, source}]` + `feed_filters` (feed báo VN theo *chuyên mục* nên phải lọc keyword để giữ đúng topic, tránh làm bẩn như top-headlines).
- Feed đang dùng (đã kiểm tra sống): VnExpress `kinh-doanh.rss` & `so-hoa.rss`, CafeF `tai-chinh-ngan-hang.rss`. Summary RSS là HTML → strip bằng BeautifulSoup. Hiện gắn cho topic `fintech` + `regulatory`.
- Kết quả sau khi thêm RSS: index có cả `vi` (VnExpress/CafeF fintech VN) lẫn `en` (GNews global); query tiếng Việt trả đúng tin fintech VN (score ~0.45). Thêm topic khác chỉ cần thêm `feeds`/`feed_filters` vào JSON.
- Nội dung sâu về đối thủ (ZaloPay/MoMo/VNPay trên site chính thức) vẫn thuộc **Sub A**; tái dùng `_bilingual_queries` cho search của Sub A.

Metadata: `topic` (fintech|ai_product|growth_marketing|regulatory), `source`, `source_url`, `article_url`, `title`, `description`, `published_at`, `language`, `crawled_at`. ID: `news_{topic}_{YYYYMMDD}_{md5(url)[:8]}`.

Tools:
- `search_industry_trends(query, topic?, from_date?, to_date?, top_k=5)` — qua Pinecone.
- `get_top_headlines(category, lang="en", max_results=10)` — realtime, không qua DB (đã có).
- `search_news_realtime(keyword, lang="en", from_date?, max_results=10)` — realtime, boolean AND/OR/NOT (đã có `search_news`).
- `get_weekly_trend_digest(topics?, days_back=7)` — tổng hợp 7 ngày.
- `trigger_crawl(topics?)` — index tin mới; Render Cron gọi 7am + 7pm.

## Environment Variables

Set trong `.env` (gitignored, KHÔNG vào Docker build vì .gitignore — trên Render set ở Dashboard). **Bắt buộc** (`src/config.py` fail-fast lúc import): `SCALEKIT_ENVIRONMENT_URL`, `SCALEKIT_CLIENT_ID`, `SCALEKIT_CLIENT_SECRET`, `SCALEKIT_RESOURCE_IDENTIFIER`, `SCALEKIT_RESOURCE_METADATA_URL`, `SCALEKIT_AUTHORIZATION_SERVERS`, `SCALEKIT_AUDIENCE_NAME`, `SCALEKIT_RESOURCE_DOCS_URL`. **Theo feature** (check lúc runtime): `PINECONE_API_KEY`, `PINECONE_INDEX_NAME`, `OPENAI_API_KEY`, `GNEWS_API_KEY`, `TAVILY_API_KEY`, `REFERO_MCP_URL`, `REFERO_TOKEN`, `RENDER_EXTERNAL_URL`, `PORT` (default 10000), `CRON_SECRET` (bảo vệ `/internal/crawl`), `DB_PATH` (override đường SQLite → trỏ vào persistent disk).

## Scheduling (Render Cron Jobs)

Crawl KHÔNG có scheduler in-process. Lịch chạy qua **Render Cron Job** gọi endpoint `POST /internal/crawl?target=<rag|news|reviews|all>` (thêm `&last_days=N` cho reviews). Endpoint này bỏ qua Scalekit, bảo vệ bằng header `X-Cron-Secret == CRON_SECRET`, và **chạy crawl NGAY TRONG web service** (background thread) → ghi vào SQLite của chính nó + Pinecone. Cron KHÔNG tự crawl (tránh ghi vào SQLite riêng của container cron).

- `cron_trigger.py` (stdlib) = lệnh cron chạy: `uv run python cron_trigger.py <target>`; đọc `WEB_SERVICE_URL` + `CRON_SECRET`.
- `render.yaml` = Blueprint 2 cron: `crawl-daily` (0 0 * * * = 7h VN, target all) và `crawl-news-pm` (0 12 * * * = 19h VN, target news). Cron dùng UTC; VN = UTC+7.
- **Set env**: `CRON_SECRET` trên CẢ web service (để validate) lẫn 2 cron job; `WEB_SERVICE_URL` trên cron job.
- **SQLite bền vững**: local SQLite là ephemeral trên Render (reset khi redeploy/spin-down; Pinecone không ảnh hưởng). Muốn digest/insights/compare/dedup bền → gắn **Persistent Disk** cho web service, mount ví dụ `/var/data`, set `DB_PATH=/var/data/competitor_intel.db` (disk chỉ gắn 1 service — hợp lý vì crawl chạy trong web service). Không có disk thì dữ liệu bookkeeping dựng lại ở lần crawl kế tiếp.
- Thay cho Render Cron có thể dùng cron ngoài (cron-job.org, GitHub Actions) POST thẳng `/internal/crawl` với header secret — không tốn build image.

## Thứ tự build đề xuất

1. Hạ tầng dùng chung: đã có `_embed` / `_get_index` / Jina Reader / dedup trong `rag_mcp.py` — tách ra dùng chung nếu cần; thêm `tavily_client`.
2. **Sub D** (gần xong nhất: chỉ thêm index + digest quanh `gnews_mcp_server.py`).
3. **Sub A** (mở rộng `rag_mcp.py` + Tavily fallback).
4. **Sub B** (App reviews, bỏ BigQuery).
5. **Sub C** (Refero MCP client).
6. Với mỗi subsystem: mount trong `main.py` + well-known route + nhánh `auth.py` (xem "Thêm 1 MCP server mới").
7. Render Cron Job gọi các `trigger_crawl` theo lịch.

## Acceptance Criteria

- Sub A: `search_competitor_strategy(...)` trả kết quả có source+date; miss → tự Tavily → lưu DB → trả kết quả.
- Sub B: `search_app_reviews("lỗi OTP", app="momo", rating_max=2)` trả review liên quan; `get_review_insights` tổng hợp top vấn đề; review crawl xong query được ngay.
- Sub C: `search_ux_patterns(...)` / `compare_user_flow(...)` trả kết quả từ Refero.
- Sub D: `search_industry_trends`, `get_top_headlines`, `search_news_realtime`, `get_weekly_trend_digest` chạy đúng.
- Chung: mọi tool qua được Auth (scope `search:read`), có error handling, và **trả lời bằng tiếng Việt/Anh** dù nguồn ngôn ngữ khác; `trigger_crawl` gọi được từ cron ngoài.
