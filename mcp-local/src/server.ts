// src/server.ts
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";

import * as fs from "fs/promises";
import * as fssync from "fs";
import * as path from "path";
import fg from "fast-glob";
import { applyPatch } from "diff";

const ROOT = path.resolve(process.env.ALLOWED_ROOT || process.cwd());
const ALLOW_CMDS = (process.env.ALLOW_CMDS || "git,npm,pnpm,yarn,dotnet").split(",");
const MAX_BYTES = Number(process.env.MAX_BYTES || 2_000_000);

function withinRoot(p: string) {
  const abs = path.resolve(ROOT, p);
  if (!abs.startsWith(ROOT)) throw new Error(`Path escapes root: ${p}`);
  return abs;
}
async function safeRead(abs: string) {
  const stat = await fs.stat(abs);
  if (stat.size > MAX_BYTES) throw new Error(`Refusing to read >${MAX_BYTES} bytes (${stat.size}): ${abs}`);
  return await fs.readFile(abs, "utf8");
}

const server = new McpServer({ name: "local-fs-mcp", version: "0.1.0" });

// list_files
server.registerTool(
  "list_files",
  {
    title: "List files (glob)",
    description: "List files by glob relative to root",
    inputSchema: {
      pattern: z.string().describe("e.g. '**/*.cs' or 'src/**'"),
      ignore: z.string().default(".git,node_modules"),
      limit: z.number().default(200)
    }
  },
  async ({ pattern, ignore, limit }) => {
    const ig = ignore.split(",").map(s => s.trim()).filter(Boolean);
    const files = await fg(pattern, { cwd: ROOT, ignore: ig, dot: true, onlyFiles: true });
    return { content: [{ type: "json", json: { ok: true, files: files.slice(0, limit) } }] };
  }
);

// read_file
server.registerTool(
  "read_file",
  {
    title: "Read file",
    description: "Read a UTF-8 text file within the root",
    inputSchema: { path: z.string() }
  },
  async ({ path: rel }) => {
    const abs = withinRoot(rel);
    const content = await safeRead(abs);
    return { content: [{ type: "text", text: content }] };
  }
);

// write_file
server.registerTool(
  "write_file",
  {
    title: "Write file",
    description: "Write a UTF-8 text file. Creates parent dirs. Makes .bak before overwrite.",
    inputSchema: {
      path: z.string(),
      content: z.string(),
      make_backup: z.boolean().default(true)
    }
  },
  async ({ path: rel, content, make_backup }) => {
    const abs = withinRoot(rel);
    await fs.mkdir(path.dirname(abs), { recursive: true });
    if (make_backup && fssync.existsSync(abs)) {
      const bak = abs + "." + Date.now() + ".bak";
      await fs.copyFile(abs, bak);
    }
    if (Buffer.byteLength(content, "utf8") > MAX_BYTES) {
      throw new Error(`Refusing to write >${MAX_BYTES} bytes`);
    }
    await fs.writeFile(abs, content, "utf8");
    return { content: [{ type: "json", json: { ok: true, path: rel } }] };
  }
);

// apply_patch (unified diff)
server.registerTool(
  "apply_patch",
  {
    title: "Apply unified diff",
    description: "Apply a unified diff patch to a file and return final content.",
    inputSchema: {
      path: z.string(),
      patch: z.string(),
      dry_run: z.boolean().default(false),
      make_backup: z.boolean().default(true)
    }
  },
  async ({ path: rel, patch, dry_run, make_backup }) => {
    const abs = withinRoot(rel);
    const original = fssync.existsSync(abs) ? await safeRead(abs) : "";
    const next = applyPatch(original, patch, { fuzzFactor: 1 });
    if (next === false) throw new Error("Patch failed to apply");
    if (!dry_run) {
      if (make_backup && fssync.existsSync(abs)) {
        const bak = abs + "." + Date.now() + ".bak";
        await fs.copyFile(abs, bak);
      }
      await fs.mkdir(path.dirname(abs), { recursive: true });
      await fs.writeFile(abs, next, "utf8");
    }
    return { content: [{ type: "text", text: next }] };
  }
);

// search_text (simple substring)
server.registerTool(
  "search_text",
  {
    title: "Search text",
    description: "Search files by substring (case-insensitive).",
    inputSchema: {
      pattern: z.string(),
      query: z.string(),
      ignore: z.string().default(".git,node_modules"),
      limit: z.number().default(200)
    }
  },
  async ({ pattern, query, ignore, limit }) => {
    const files = await fg(pattern, { cwd: ROOT, ignore: ignore.split(","), dot: true, onlyFiles: true });
    const needle = String(query).toLowerCase();
    const hits: Array<{ path: string; line: number; preview: string }> = [];
    for (const rel of files) {
      const abs = withinRoot(rel);
      const text = await safeRead(abs);
      const lines = text.split(/\r?\n/);
      for (let i = 0; i < lines.length; i++) {
        if (lines[i].toLowerCase().includes(needle)) {
          hits.push({ path: rel, line: i + 1, preview: lines[i].slice(0, 240) });
          if (hits.length >= limit) break;
        }
      }
      if (hits.length >= limit) break;
    }
    return { content: [{ type: "json", json: { ok: true, hits } }] };
  }
);

// run_command (allow-list)
server.registerTool(
  "run_command",
  {
    title: "Run command (allow-listed)",
    description: "Run an allow-listed CLI in repo root (git, npm, pnpm, yarn, dotnet).",
    inputSchema: {
      cmd: z.string(),
      args: z.array(z.string()).default([]),
      timeout_ms: z.number().default(60000)
    }
  },
  async ({ cmd, args, timeout_ms }) => {
    if (!ALLOW_CMDS.includes(cmd)) throw new Error(`Command not allowed: ${cmd}`);
    const { execa } = await import("execa");
    const subprocess = execa(cmd, args, { cwd: ROOT, timeout: timeout_ms });
    const { stdout, stderr, exitCode } = await subprocess;
    return { content: [{ type: "json", json: { ok: exitCode === 0, exitCode, stdout, stderr } }] };
  }
);

async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error(`[MCP] Started. Root=${ROOT} Allowed=${ALLOW_CMDS.join(",")}`);
}

main().catch((e) => {
  console.error("Server error:", e);
  process.exit(1);
});
