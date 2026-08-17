// Package tclaudesessions enumerates the tclaude sessions recorded for a
// working directory — the Tencent wrapper around Claude Code.
//
// tclaude (npm @tencent/tclaude) forwards every arg to upstream Claude Code
// but sets CLAUDE_CONFIG_DIR=~/.tclaude internally, so its transcripts live at
// ~/.tclaude/projects/<escaped-cwd>/<session-id>.jsonl. The directory layout
// and the EscapeCWD algorithm are identical to Claude Code's (see
// internal/claudesessions), so this package reuses claudesessions.Session /
// claudesessions.Detail and claudesessions.EscapeCWD rather than redefining
// them. What differs is the transcript FORMAT:
//
//   - tclaude transcripts have no "promptSource":"typed" field. A typed human
//     prompt is a "type":"user" entry whose message.content is a PLAIN STRING
//     (a tool result is also "type":"user" but carries content as an ARRAY —
//     that distinction is what separates a real prompt from a tool result).
//   - tclaude writes "type":"ai-title" entries carrying an aiTitle field, which
//     is a cleaner session title than the first typed prompt. We prefer it and
//     fall back to the first string-content user entry when none is present in
//     the scanned head.
//
// Pure reads, no process spawning, no network I/O: every failure degrades to
// "fewer sessions" rather than an error, so discovery can never block pane
// creation. Titles are sanitized of control characters — a transcript records
// whatever the user typed, and the value is rendered into a TUI.
package tclaudesessions

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
	"unicode"

	"github.com/artyomsv/quil/internal/claudesessions"
)

// Reuse the public types from the claude package — the on-disk Session/Detail
// shape is identical, and the daemon's IPC carries claudesessions.Session
// values, so returning the same type avoids a conversion. Only the config-dir
// name and transcript format differ.
type Session = claudesessions.Session
type Detail = claudesessions.Detail

const (
	MaxSessions    = 200
	MaxTitleRunes  = 240
	MaxPromptRunes = 1200
	// titleScanBytes caps how much of a transcript head is read while looking
	// for an ai-title or first prompt.
	titleScanBytes = 64 << 10
	// startScanLines bounds the search for the session's start timestamp.
	startScanLines = 20
)

// EscapeCWD is the same per-project directory naming Claude Code uses; tclaude
// shares it. Delegated so a future fix in one place fixes both.
func EscapeCWD(cwd string) string { return claudesessions.EscapeCWD(cwd) }

// ProjectDir returns the absolute directory tclaude stores this CWD's session
// transcripts in, or "" when the user's home directory cannot be resolved.
func ProjectDir(cwd string) string {
	if cwd == "" {
		return ""
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return ""
	}
	return filepath.Join(home, ".tclaude", "projects", EscapeCWD(cwd))
}

// TranscriptPath returns the absolute path of one session's transcript, or ""
// when the home directory is unavailable or either argument is empty.
func TranscriptPath(cwd, sessionID string) string {
	dir := ProjectDir(cwd)
	if dir == "" || sessionID == "" {
		return ""
	}
	return filepath.Join(dir, sessionID+".jsonl")
}

// List returns the tclaude sessions recorded for cwd, newest first, capped at
// MaxSessions. See claudesessions.List for the failure-mode contract — it is
// identical here.
func List(ctx context.Context, cwd string) (sessions []Session, truncated bool, err error) {
	dir := ProjectDir(cwd)
	if dir == "" || ctx.Err() != nil {
		return nil, false, nil
	}
	return listDir(ctx, dir)
}

func listDir(ctx context.Context, dir string) (sessions []Session, truncated bool, err error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, false, nil
		}
		return nil, false, err
	}

	type candidate struct {
		id   string
		path string
		info os.FileInfo
	}
	candidates := make([]candidate, 0, len(entries))
	for _, e := range entries {
		if ctx.Err() != nil {
			return nil, false, nil
		}
		if !strings.HasSuffix(e.Name(), ".jsonl") {
			continue
		}
		info, err := e.Info()
		if err != nil || !info.Mode().IsRegular() || info.Size() == 0 {
			continue
		}
		candidates = append(candidates, candidate{
			id:   strings.TrimSuffix(e.Name(), ".jsonl"),
			path: filepath.Join(dir, e.Name()),
			info: info,
		})
	}

	sort.Slice(candidates, func(i, j int) bool {
		ti, tj := candidates[i].info.ModTime(), candidates[j].info.ModTime()
		if ti.Equal(tj) {
			return candidates[i].id < candidates[j].id
		}
		return ti.After(tj)
	})
	if len(candidates) > MaxSessions {
		candidates = candidates[:MaxSessions]
		truncated = true
	}

	out := make([]Session, 0, len(candidates))
	for _, c := range candidates {
		if ctx.Err() != nil {
			return out, truncated, nil
		}
		out = append(out, Session{
			ID:       c.id,
			Title:    readTitle(c.path),
			Modified: c.info.ModTime(),
		})
	}
	return out, truncated, nil
}

// transcriptLine mirrors the subset of a tclaude transcript entry needed to
// recognize a typed human prompt and pull its text. Extra fields are ignored.
type transcriptLine struct {
	Type        string `json:"type"`
	IsSidechain bool   `json:"isSidechain"`
	Timestamp   string `json:"timestamp"`
	AiTitle     string `json:"aiTitle"`
	Message     struct {
		// Content is a string for a plain prompt and an array of content
		// blocks for a tool result — both "type":"user" shapes occur, so it is
		// decoded in a second pass and string-vs-array distinguished by
		// contentIsString.
		Content json.RawMessage `json:"content"`
	} `json:"message"`
}

// ReadDetail reads one tclaude session's transcript for cwd and summarizes it.
// See claudesessions.ReadDetail for the validation + cancellation contract.
func ReadDetail(ctx context.Context, cwd, sessionID string) (Detail, error) {
	if err := ctx.Err(); err != nil {
		return Detail{}, fmt.Errorf("read session detail: %w", err)
	}
	if sessionID == "" || sessionID != filepath.Base(sessionID) ||
		sessionID == "." || sessionID == ".." || strings.ContainsRune(sessionID, filepath.Separator) {
		return Detail{}, fmt.Errorf("invalid session id")
	}
	path := TranscriptPath(cwd, sessionID)
	if path == "" {
		return Detail{}, fmt.Errorf("no transcript path for this session")
	}
	return readDetail(ctx, path, sessionID)
}

func readDetail(ctx context.Context, path, id string) (Detail, error) {
	info, err := os.Lstat(path)
	if err != nil {
		return Detail{}, err
	}
	if !info.Mode().IsRegular() {
		return Detail{}, fmt.Errorf("transcript is not a regular file")
	}

	f, err := os.Open(path)
	if err != nil {
		return Detail{}, err
	}
	defer f.Close()

	d := Detail{ID: id, Modified: info.ModTime(), SizeBytes: info.Size()}

	// tclaude tool results are "type":"user" with array content; the byte
	// pre-filter admits them, so consume rejects array-content lines as tool
	// results rather than counting them as prompts. Cheap byte-compare before
	// any JSON parsing keeps the scan affordable on multi-MB transcripts.
	r := bufio.NewReaderSize(f, 64<<10)
	userMark := []byte(`"type":"user"`)
	for lineNo := 0; ; lineNo++ {
		if lineNo%4096 == 0 {
			if err := ctx.Err(); err != nil {
				return Detail{}, fmt.Errorf("scan transcript %s at line %d: %w", id, lineNo, err)
			}
		}
		line, readErr := r.ReadBytes('\n')
		if len(line) > 0 {
			consumeDetailLine(&d, line, lineNo, userMark)
		}
		if readErr != nil {
			break
		}
	}
	return d, nil
}

// consumeDetailLine folds one transcript line into the accumulating detail.
// A standalone function (not a Detail method) because Detail is an alias for
// claudesessions.Detail, whose own consume is unexported — this package cannot
// define methods on the aliased type that the alias's home package did not.
func consumeDetailLine(d *Detail, line []byte, lineNo int, userMark []byte) {
	needStart := d.Started.IsZero() && lineNo < startScanLines

	if !needStart && !bytes.Contains(line, userMark) {
		return
	}

	var tl transcriptLine
	if err := json.Unmarshal(bytes.TrimSpace(line), &tl); err != nil {
		return
	}
	if needStart && tl.Timestamp != "" {
		if ts, err := time.Parse(time.RFC3339, tl.Timestamp); err == nil {
			d.Started = ts
		}
	}
	// Sidechain entries belong to subagents, not the user's conversation.
	// A real typed prompt is type:user with STRING content; array content is a
	// tool result (also type:user in tclaude) and must not be counted.
	if tl.IsSidechain || tl.Type != "user" || !contentIsString(tl.Message.Content) {
		return
	}
	d.UserPrompts++
	if text := sanitizePrompt(contentText(tl.Message.Content, MaxPromptRunes)); text != "" {
		if d.FirstPrompt == "" {
			d.FirstPrompt = text
		}
		d.LastPrompt = text
	}
}

// readTitle returns the session's ai-title if one sits in the transcript head,
// falling back to the first typed human prompt. Any failure yields "".
func readTitle(path string) string {
	f, err := os.Open(path)
	if err != nil {
		return ""
	}
	defer f.Close()

	buf, err := io.ReadAll(io.LimitReader(f, titleScanBytes))
	if err != nil || len(buf) == 0 {
		return ""
	}
	lines := bytes.Split(buf, []byte{'\n'})
	if len(buf) == titleScanBytes && buf[len(buf)-1] != '\n' && len(lines) > 0 {
		lines = lines[:len(lines)-1]
	}

	// First pass: prefer an ai-title entry — it is a cleaner, model-generated
	// summary than the raw first prompt.
	aiTitleMark := []byte(`"type":"ai-title"`)
	for _, line := range lines {
		if !bytes.Contains(line, aiTitleMark) {
			continue
		}
		var tl transcriptLine
		if err := json.Unmarshal(bytes.TrimSpace(line), &tl); err != nil {
			continue
		}
		if text := sanitizeTitle(tl.AiTitle); text != "" {
			return text
		}
	}

	// Fallback: the first typed human prompt (string content, not a tool
	// result, not a sidechain).
	userMark := []byte(`"type":"user"`)
	for _, line := range lines {
		if !bytes.Contains(line, userMark) {
			continue
		}
		var tl transcriptLine
		if err := json.Unmarshal(bytes.TrimSpace(line), &tl); err != nil {
			continue
		}
		if tl.Type != "user" || tl.IsSidechain || !contentIsString(tl.Message.Content) {
			continue
		}
		if text := sanitizeTitle(contentText(tl.Message.Content, MaxTitleRunes)); text != "" {
			return text
		}
	}
	return ""
}

// contentIsString reports whether a message.content raw JSON value decodes as a
// plain string (a typed prompt) rather than an array of content blocks (a tool
// result). tclaude records both as type:"user", so this is the discriminator.
func contentIsString(raw json.RawMessage) bool {
	if len(raw) == 0 {
		return false
	}
	var s string
	return json.Unmarshal(raw, &s) == nil
}

// contentText flattens a message content field into plain text. A typed prompt
// is a bare string; a tool result is an array and returns "" (the caller has
// already filtered those out via contentIsString, but this stays defensive).
func contentText(raw json.RawMessage, limit int) string {
	if len(raw) == 0 {
		return ""
	}
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		return s
	}
	var blocks []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}
	if err := json.Unmarshal(raw, &blocks); err != nil {
		return ""
	}
	var b strings.Builder
	for _, blk := range blocks {
		if blk.Type != "text" || blk.Text == "" {
			continue
		}
		if b.Len() > 0 {
			b.WriteByte(' ')
		}
		b.WriteString(blk.Text)
		if b.Len() >= limit {
			break
		}
	}
	return b.String()
}

// sanitizeTitle and sanitizePrompt mirror claudesessions' sanitizers: collapse
// control/format characters so a pasted terminal capture or bidi-override
// cannot spoof the TUI. Local copies because the originals are unexported.
func sanitizeTitle(s string) string {
	mapped := strings.Map(func(r rune) rune {
		switch {
		case r == '\n' || r == '\r' || r == '\t':
			return ' '
		case unicode.IsControl(r), unicode.Is(unicode.Cf, r):
			return -1
		default:
			return r
		}
	}, s)
	title := strings.Join(strings.Fields(mapped), " ")

	runes := []rune(title)
	if len(runes) > MaxTitleRunes {
		title = strings.TrimSpace(string(runes[:MaxTitleRunes])) + "…"
	}
	return title
}

func sanitizePrompt(s string) string {
	mapped := strings.Map(func(r rune) rune {
		switch {
		case r == '\n':
			return r
		case r == '\r':
			return -1
		case r == '\t':
			return ' '
		case unicode.IsControl(r), unicode.Is(unicode.Cf, r):
			return -1
		default:
			return r
		}
	}, s)

	var out []string
	blank := 0
	for _, line := range strings.Split(mapped, "\n") {
		line = strings.TrimRight(line, " \t")
		if line == "" {
			blank++
			if blank > 1 || len(out) == 0 {
				continue
			}
		} else {
			blank = 0
		}
		out = append(out, line)
	}
	text := strings.TrimRight(strings.Join(out, "\n"), "\n")

	runes := []rune(text)
	if len(runes) > MaxPromptRunes {
		text = strings.TrimRight(string(runes[:MaxPromptRunes]), " \n") + "…"
	}
	return text
}
