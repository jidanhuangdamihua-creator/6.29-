import fs from "node:fs/promises";
import { importRuntimeModule } from "/Users/ming/.codex/plugins/cache/openai-primary-runtime/presentations/26.904.11930/skills/presentations/container_tools/runtime_helpers.mjs";

const { SpreadsheetFile, Workbook } = await importRuntimeModule("@oai/artifact-tool");

const [, , inputPath, outputPath] = process.argv;
if (!inputPath || !outputPath) {
  throw new Error("usage: write_bl6_bl7_candidate_xlsx.mjs <payload.json> <output.xlsx>");
}

const payload = JSON.parse(await fs.readFile(inputPath, "utf8"));
const workbook = Workbook.create();
const sheet = workbook.worksheets.add("BL6_BL7_Formal_Candidate");
const matrix = [
  payload.columns,
  ...payload.rows.map((row) => Array.isArray(row)
    ? row
    : payload.columns.map((column) => row[column] ?? null)),
];
const endColumn = (() => {
  let n = payload.columns.length;
  let label = "";
  while (n > 0) {
    const rem = (n - 1) % 26;
    label = String.fromCharCode(65 + rem) + label;
    n = Math.floor((n - 1) / 26);
  }
  return label;
})();
const endRow = matrix.length;
const range = sheet.getRange(`A1:${endColumn}${endRow}`);
range.values = matrix;
const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);
