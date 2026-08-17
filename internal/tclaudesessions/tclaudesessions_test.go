package tclaudesessions

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
	"unicode/utf8"
)

// TestEscapeCWD delegates to claudesessions and the algorithm is shared, so we
// only sanity-check the field-evidence vector that matters for this machine
// rather than re-running claude's full table. If claude's escaper changes,
// claude's own TestEscapeCWD fails first.
func TestEscapeCWD_DelegatesToClaude(t *testing.T) {
	if got, want := EscapeCWD(`F:\Release_4_1`), "F--Release-4-1"; got != want {
		t.Errorf("EscapeCWD = %q, want %q", got, want)
	}
	if got, want := EscapeCWD(""), ""; got != want {
		t.Errorf("EscapeCWD(\"\") = %q, want empty", got)
	}
}

func TestProjectDir_TargetsTclaude(t *testing.T) {
	if got := ProjectDir(""); got != "" {
		t.Errorf("ProjectDir(\"\") = %q, want empty", got)
	}
	got := ProjectDir("/home/user/proj")
	if got == "" {
		t.Skip("home directory unavailable in this environment")
	}
	// The config-dir discriminator: tclaude, NOT claude.
	if !strings.Contains(got, ".tclaude") {
		t.Errorf("ProjectDir = %q, want it under .tclaude", got)
	}
	if strings.Contains(got, ".claude"+string(filepath.Separator)+"projects") && !strings.Contains(got, ".tclaude") {
		t.Errorf("ProjectDir = %q landed under .claude, not .tclaude", got)
	}
}

func TestTranscriptPath_EmptySessionID_ReturnsEmpty(t *testing.T) {
	if got := TranscriptPath("/home/user/proj", ""); got != "" {
		t.Errorf("TranscriptPath with empty id = %q, want empty", got)
	}
}

func TestTranscriptPath_BuildsJSONLPath(t *testing.T) {
	got := TranscriptPath("/home/user/proj", "abc-123")
	if got == "" {
		t.Skip("home directory unavailable in this environment")
	}
	wantSuffix := filepath.Join("-home-user-proj", "abc-123.jsonl")
	if !strings.HasSuffix(got, wantSuffix) {
		t.Errorf("TranscriptPath = %q, want suffix %q", got, wantSuffix)
	}
	if !strings.Contains(got, ".tclaude") {
		t.Errorf("TranscriptPath = %q, want it under .tclaude", got)
	}
}

// --- tclaude transcript fixture builders -----------------------------------
//
// tclaude's transcript differs from claude's in two ways that this package
// exists to handle:
//   - typed prompts are "type":"user" with STRING content and NO promptSource;
//   - tool results are "type":"user" with ARRAY content (a content-block list);
//   - session titles come from "type":"ai-title" entries.

func typedPrompt(text string) string {
	return fmt.Sprintf(
		`{"type":"user","isSidechain":false,"message":{"role":"user","content":%q},"timestamp":"2026-07-01T10:00:00.000Z"}`,
		text)
}

// toolResult is a "type":"user" entry whose content is an ARRAY — the shape
// tclaude records for a tool result. This is the discriminator: a string
// content is a prompt, an array content is not.
func toolResult(text string) string {
	return fmt.Sprintf(
		`{"type":"user","isSidechain":false,"message":{"role":"user","content":[{"type":"text","text":%q}]},"timestamp":"2026-07-01T10:05:00.000Z"}`,
		text)
}

func assistantMsg(text string) string {
	return fmt.Sprintf(
		`{"type":"assistant","isSidechain":false,"message":{"role":"assistant","content":%q},"timestamp":"2026-07-01T10:01:00.000Z"}`,
		text)
}

func aiTitle(title string) string {
	return fmt.Sprintf(`{"type":"ai-title","aiTitle":%q,"leafUuid":"u1","timestamp":"2026-07-01T10:00:30.000Z"}`, title)
}

func writeSession(t *testing.T, dir, id string, mtime time.Time, lines ...string) string {
	t.Helper()
	path := filepath.Join(dir, id+".jsonl")
	body := strings.Join(lines, "\n")
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatalf("write session %s: %v", id, err)
	}
	if err := os.Chtimes(path, mtime, mtime); err != nil {
		t.Fatalf("chtimes %s: %v", id, err)
	}
	return id
}

func TestListDir_MissingDirectory_ReturnsEmptyNotError(t *testing.T) {
	got, _, err := listDir(context.Background(), filepath.Join(t.TempDir(), "does-not-exist"))
	if err != nil {
		t.Fatalf("listDir on missing dir returned error: %v", err)
	}
	if len(got) != 0 {
		t.Errorf("listDir on missing dir = %d sessions, want 0", len(got))
	}
}

func TestListDir_SortsNewestFirst(t *testing.T) {
	dir := t.TempDir()
	base := time.Date(2026, 7, 1, 12, 0, 0, 0, time.UTC)
	writeSession(t, dir, "oldest", base.Add(-48*time.Hour), aiTitle("old task"))
	writeSession(t, dir, "newest", base, aiTitle("new task"))
	writeSession(t, dir, "middle", base.Add(-24*time.Hour), aiTitle("mid task"))

	got, _, err := listDir(context.Background(), dir)
	if err != nil {
		t.Fatalf("listDir: %v", err)
	}
	want := []string{"newest", "middle", "oldest"}
	if len(got) != len(want) {
		t.Fatalf("listDir returned %d sessions, want %d", len(got), len(want))
	}
	for i, id := range want {
		if got[i].ID != id {
			t.Errorf("session[%d].ID = %q, want %q", i, got[i].ID, id)
		}
	}
}

// TestListDir_PrefersAiTitle: the ai-title entry is a cleaner summary than the
// raw first prompt, so it wins when present.
func TestListDir_PrefersAiTitle(t *testing.T) {
	dir := t.TempDir()
	writeSession(t, dir, "s1", time.Now(),
		`{"type":"mode","mode":"normal","sessionId":"s1"}`,
		typedPrompt("Add resume option to the setup dialog"),
		aiTitle("Resume picker for tclaude"),
		typedPrompt("a later prompt that must not win"),
	)

	got, _, err := listDir(context.Background(), dir)
	if err != nil {
		t.Fatalf("listDir: %v", err)
	}
	if len(got) != 1 {
		t.Fatalf("listDir returned %d sessions, want 1", len(got))
	}
	if want := "Resume picker for tclaude"; got[0].Title != want {
		t.Errorf("Title = %q, want the ai-title %q", got[0].Title, want)
	}
}

// TestListDir_FallsBackToFirstPromptWhenNoAiTitle: when no ai-title sits in the
// scanned head, the first typed prompt (string content) is the title.
func TestListDir_FallsBackToFirstPromptWhenNoAiTitle(t *testing.T) {
	dir := t.TempDir()
	writeSession(t, dir, "s1", time.Now(),
		`{"type":"mode","mode":"normal","sessionId":"s1"}`,
		typedPrompt("the first real prompt"),
		typedPrompt("a later prompt that must not win"),
	)

	got, _, err := listDir(context.Background(), dir)
	if err != nil {
		t.Fatalf("listDir: %v", err)
	}
	if want := "the first real prompt"; got[0].Title != want {
		t.Errorf("Title = %q, want first prompt fallback %q", got[0].Title, want)
	}
}

// TestListDir_ToolResultsAreNotPrompts: a tool result is type:"user" with ARRAY
// content — it must never be used as the title or counted as a prompt. This is
// the core tclaude-specific discrimination.
func TestListDir_ToolResultsAreNotPrompts(t *testing.T) {
	dir := t.TempDir()
	writeSession(t, dir, "s1", time.Now(),
		// Tool result appears BEFORE the real prompt; if the reader mistook it
		// for a prompt the title would be the tool output, not the user's words.
		toolResult("tool output that must not become the title"),
		typedPrompt("the real first prompt"),
	)

	got, _, err := listDir(context.Background(), dir)
	if err != nil {
		t.Fatalf("listDir: %v", err)
	}
	if want := "the real first prompt"; got[0].Title != want {
		t.Errorf("Title = %q, want %q (tool result with array content was treated as a prompt)", got[0].Title, want)
	}
}

func TestListDir_SkipsSidechainPrompts(t *testing.T) {
	dir := t.TempDir()
	writeSession(t, dir, "s1", time.Now(),
		`{"type":"user","isSidechain":true,"message":{"role":"user","content":"subagent instruction"}}`,
		typedPrompt("the real first prompt"),
	)

	got, _, err := listDir(context.Background(), dir)
	if err != nil {
		t.Fatalf("listDir: %v", err)
	}
	if want := "the real first prompt"; got[0].Title != want {
		t.Errorf("Title = %q, want %q (sidechain leaked into the title)", got[0].Title, want)
	}
}

func TestListDir_MalformedLinesSkipped(t *testing.T) {
	dir := t.TempDir()
	writeSession(t, dir, "s1", time.Now(),
		`{"type":"ai-title" this is not json`,
		typedPrompt("valid prompt"),
	)

	got, _, err := listDir(context.Background(), dir)
	if err != nil {
		t.Fatalf("listDir: %v", err)
	}
	if want := "valid prompt"; got[0].Title != want {
		t.Errorf("Title = %q, want %q", got[0].Title, want)
	}
}

func TestListDir_NoPromptOrTitle_EmptyTitleStillListed(t *testing.T) {
	dir := t.TempDir()
	writeSession(t, dir, "s1", time.Now(),
		`{"type":"mode","mode":"normal","sessionId":"s1"}`,
	)

	got, _, err := listDir(context.Background(), dir)
	if err != nil {
		t.Fatalf("listDir: %v", err)
	}
	if len(got) != 1 {
		t.Fatalf("listDir returned %d sessions, want 1 (a titleless session must still list)", len(got))
	}
	if got[0].Title != "" {
		t.Errorf("Title = %q, want empty", got[0].Title)
	}
}

func TestListDir_IgnoresDirectoriesAndNonJSONL(t *testing.T) {
	dir := t.TempDir()
	writeSession(t, dir, "real", time.Now(), aiTitle("real session"))
	if err := os.Mkdir(filepath.Join(dir, "scratch.jsonl"), 0o755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	if err := os.WriteFile(filepath.Join(dir, "notes.txt"), []byte("x"), 0o644); err != nil {
		t.Fatalf("write: %v", err)
	}

	got, _, err := listDir(context.Background(), dir)
	if err != nil {
		t.Fatalf("listDir: %v", err)
	}
	if len(got) != 1 || got[0].ID != "real" {
		t.Errorf("listDir = %+v, want only the real session", got)
	}
}

func TestListDir_SkipsEmptyFiles(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "empty.jsonl"), nil, 0o644); err != nil {
		t.Fatalf("write: %v", err)
	}

	got, _, err := listDir(context.Background(), dir)
	if err != nil {
		t.Fatalf("listDir: %v", err)
	}
	if len(got) != 0 {
		t.Errorf("listDir = %d sessions, want 0", len(got))
	}
}

func TestListDir_CapsAtMaxSessions(t *testing.T) {
	dir := t.TempDir()
	base := time.Date(2026, 7, 1, 12, 0, 0, 0, time.UTC)
	for i := 0; i < MaxSessions+25; i++ {
		writeSession(t, dir, fmt.Sprintf("s%03d", i),
			base.Add(time.Duration(i)*time.Minute), aiTitle("prompt"))
	}

	got, truncated, err := listDir(context.Background(), dir)
	if err != nil {
		t.Fatalf("listDir: %v", err)
	}
	if len(got) != MaxSessions {
		t.Fatalf("listDir returned %d sessions, want %d", len(got), MaxSessions)
	}
	if !truncated {
		t.Error("truncated = false, want true when sessions were dropped by the cap")
	}
	if want := fmt.Sprintf("s%03d", MaxSessions+24); got[0].ID != want {
		t.Errorf("got[0].ID = %q, want %q (newest must survive the cap)", got[0].ID, want)
	}
}

func TestListDir_ExactlyCapIsNotTruncated(t *testing.T) {
	dir := t.TempDir()
	base := time.Date(2026, 7, 1, 12, 0, 0, 0, time.UTC)
	for i := 0; i < MaxSessions; i++ {
		writeSession(t, dir, fmt.Sprintf("s%03d", i),
			base.Add(time.Duration(i)*time.Minute), aiTitle("prompt"))
	}

	got, truncated, err := listDir(context.Background(), dir)
	if err != nil {
		t.Fatalf("listDir: %v", err)
	}
	if len(got) != MaxSessions {
		t.Fatalf("listDir returned %d sessions, want %d", len(got), MaxSessions)
	}
	if truncated {
		t.Error("truncated = true for exactly the cap with nothing dropped")
	}
}

// --- Sanitizers (mirror claude's behavior) --------------------------------

func TestSanitizeTitle(t *testing.T) {
	tests := []struct {
		name string
		in   string
		want string
	}{
		{"plain passes through", "fix the daemon", "fix the daemon"},
		{"newlines collapse to spaces", "line one\nline two", "line one line two"},
		{"tabs and runs collapse", "a\t\t  b", "a b"},
		{"ansi escape dropped, not spaced", "red \x1b[31mtext\x1b[0m", "red [31mtext[0m"},
		{"null byte dropped", "a\x00b", "ab"},
		{"carriage return separates", "a\r\nb", "a b"},
		{"surrounding space trimmed", "  padded  ", "padded"},
		{"empty stays empty", "", ""},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := sanitizeTitle(tt.in); got != tt.want {
				t.Errorf("sanitizeTitle(%q) = %q, want %q", tt.in, got, tt.want)
			}
		})
	}
}

func TestSanitizeTitle_TruncatesOnRuneBoundary(t *testing.T) {
	in := strings.Repeat("é", MaxTitleRunes+50)
	got := sanitizeTitle(in)
	runes := []rune(got)
	if len(runes) != MaxTitleRunes+1 {
		t.Fatalf("sanitizeTitle produced %d runes, want %d", len(runes), MaxTitleRunes+1)
	}
	if runes[len(runes)-1] != '…' {
		t.Errorf("truncated title does not end in an ellipsis: %q", string(runes[len(runes)-3:]))
	}
	if strings.ContainsRune(got, '�') {
		t.Error("truncation split a multi-byte rune")
	}
}

// --- ReadDetail -----------------------------------------------------------

// TestReadDetail_CountsOnlyStringPrompts: a tool result is type:"user" with
// array content. Counting every type:"user" would report tool results as
// prompts; only STRING-content user entries count.
func TestReadDetail_CountsOnlyStringPrompts(t *testing.T) {
	dir := t.TempDir()
	writeSession(t, dir, "s1", time.Now(),
		typedPrompt("first thing"),
		assistantMsg("working on it"),
		toolResult("tool output that is not a prompt"),
		toolResult("more tool output"),
		typedPrompt("second thing"),
		assistantMsg("done"),
		typedPrompt("last thing"),
	)

	d, err := readDetail(context.Background(), filepath.Join(dir, "s1.jsonl"), "s1")
	if err != nil {
		t.Fatalf("readDetail: %v", err)
	}
	if d.UserPrompts != 3 {
		t.Errorf("UserPrompts = %d, want 3 (tool results are type \"user\" but have array content)", d.UserPrompts)
	}
	if d.FirstPrompt != "first thing" {
		t.Errorf("FirstPrompt = %q, want %q", d.FirstPrompt, "first thing")
	}
	if d.LastPrompt != "last thing" {
		t.Errorf("LastPrompt = %q, want %q", d.LastPrompt, "last thing")
	}
}

func TestReadDetail_SkipsSidechainPrompts(t *testing.T) {
	dir := t.TempDir()
	sidechain := `{"type":"user","isSidechain":true,"message":{"role":"user","content":"subagent task"},"timestamp":"2026-07-01T10:02:00.000Z"}`
	writeSession(t, dir, "s1", time.Now(), typedPrompt("mine"), sidechain)

	d, err := readDetail(context.Background(), filepath.Join(dir, "s1.jsonl"), "s1")
	if err != nil {
		t.Fatalf("readDetail: %v", err)
	}
	if d.UserPrompts != 1 {
		t.Errorf("UserPrompts = %d, want 1", d.UserPrompts)
	}
	if d.LastPrompt != "mine" {
		t.Errorf("LastPrompt = %q, want the user's own prompt", d.LastPrompt)
	}
}

func TestReadDetail_ReportsStartedSizeAndModified(t *testing.T) {
	dir := t.TempDir()
	mtime := time.Now().Add(-3 * time.Hour).Truncate(time.Second)
	writeSession(t, dir, "s1", mtime, typedPrompt("hello"))

	d, err := readDetail(context.Background(), filepath.Join(dir, "s1.jsonl"), "s1")
	if err != nil {
		t.Fatalf("readDetail: %v", err)
	}
	if d.Started.IsZero() {
		t.Error("Started is zero — the opening entry carried a timestamp")
	}
	if !d.Modified.Equal(mtime) {
		t.Errorf("Modified = %v, want %v", d.Modified, mtime)
	}
	if d.SizeBytes <= 0 {
		t.Errorf("SizeBytes = %d, want the file size", d.SizeBytes)
	}
	if d.ID != "s1" {
		t.Errorf("ID = %q, want s1", d.ID)
	}
}

func TestReadDetail_StartTimestampBeyondOpeningLines(t *testing.T) {
	dir := t.TempDir()
	lines := make([]string, 0, startScanLines+2)
	for i := 0; i < startScanLines+1; i++ {
		lines = append(lines, `{"type":"system","note":"no timestamp here"}`)
	}
	lines = append(lines, typedPrompt("late"))
	writeSession(t, dir, "s1", time.Now(), lines...)

	d, err := readDetail(context.Background(), filepath.Join(dir, "s1.jsonl"), "s1")
	if err != nil {
		t.Fatalf("readDetail: %v", err)
	}
	if !d.Started.IsZero() {
		t.Errorf("Started = %v, want zero past the opening-line budget", d.Started)
	}
	if d.UserPrompts != 1 {
		t.Errorf("UserPrompts = %d, want 1", d.UserPrompts)
	}
}

func TestReadDetail_MalformedAndPartialLinesSkipped(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "s1.jsonl")
	body := typedPrompt("good") + "\n" +
		`{"type":"user","message":{"role":"user","content":` + "\n" +
		`{"type":"user","isSidechain":false,"message":{"role":"user","content":"trunc`
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatalf("write: %v", err)
	}

	d, err := readDetail(context.Background(), path, "s1")
	if err != nil {
		t.Fatalf("readDetail: %v", err)
	}
	if d.UserPrompts != 1 || d.LastPrompt != "good" {
		t.Errorf("UserPrompts = %d, LastPrompt = %q; want 1 and %q — malformed lines must be skipped, not counted",
			d.UserPrompts, d.LastPrompt, "good")
	}
}

func TestReadDetail_MissingFileIsAnError(t *testing.T) {
	if _, err := readDetail(context.Background(), filepath.Join(t.TempDir(), "nope.jsonl"), "nope"); err == nil {
		t.Error("readDetail on a missing transcript returned nil error")
	}
}

func TestReadDetail_RejectsTraversalSessionID(t *testing.T) {
	for _, id := range []string{
		"",
		"..",
		".",
		"../secrets",
		"sub/dir",
		`..\windows`,
	} {
		if _, err := ReadDetail(context.Background(), t.TempDir(), id); err == nil {
			t.Errorf("ReadDetail accepted session id %q", id)
		}
	}
}

func TestSanitizePrompt(t *testing.T) {
	tests := []struct {
		name string
		in   string
		want string
	}{
		{"keeps line structure", "line one\nline two", "line one\nline two"},
		{"crlf collapses to lf", "line one\r\nline two", "line one\nline two"},
		{"tab becomes space", "a\tb", "a b"},
		{"drops escape sequences", "red \x1b[31mtext\x1b[0m", "red [31mtext[0m"},
		{"drops bidi override", "safe‮txet", "safetxet"},
		{"collapses blank runs", "a\n\n\n\nb", "a\n\nb"},
		{"drops leading blanks", "\n\n\nhello", "hello"},
		{"trims trailing blanks", "hello\n\n\n", "hello"},
		{"strips trailing spaces", "hello   \nworld  ", "hello\nworld"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := sanitizePrompt(tt.in); got != tt.want {
				t.Errorf("sanitizePrompt(%q) = %q, want %q", tt.in, got, tt.want)
			}
		})
	}
}

func TestSanitizePrompt_TruncatesOnRuneBoundary(t *testing.T) {
	in := strings.Repeat("é", MaxPromptRunes+50)
	got := sanitizePrompt(in)
	runes := []rune(got)
	if len(runes) != MaxPromptRunes+1 {
		t.Errorf("len = %d runes, want %d", len(runes), MaxPromptRunes+1)
	}
	if runes[len(runes)-1] != '…' {
		t.Errorf("truncated prompt must end with an ellipsis, got %q", string(runes[len(runes)-1]))
	}
	if !utf8.ValidString(got) {
		t.Error("truncation split a multi-byte rune")
	}
}

func TestSanitizePrompt_MultilineSurvivesTitleCollapse(t *testing.T) {
	in := "first line\nsecond line"
	if got := sanitizeTitle(in); strings.Contains(got, "\n") {
		t.Errorf("sanitizeTitle kept a newline: %q", got)
	}
	if got := sanitizePrompt(in); !strings.Contains(got, "\n") {
		t.Errorf("sanitizePrompt dropped the line break: %q", got)
	}
}

// --- Cancellation contracts (mirror claude) -------------------------------

func TestList_CancelledContext_DegradesWithoutError(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	sessions, truncated, err := List(ctx, t.TempDir())
	if err != nil {
		t.Fatalf("List returned an error on cancel: %v (want degraded, not failed)", err)
	}
	if len(sessions) != 0 {
		t.Errorf("sessions = %d, want 0", len(sessions))
	}
	if truncated {
		t.Error("truncated set on a cancelled scan")
	}
}

func TestReadDetail_CancelledContext_ReturnsError(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, err := ReadDetail(ctx, t.TempDir(), "aaaaaaaa-0000-4000-8000-000000000000")
	if err == nil {
		t.Fatal("ReadDetail succeeded on a cancelled context, want an error")
	}
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v, want it to wrap context.Canceled", err)
	}
}
