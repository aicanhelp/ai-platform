import fs from "fs";
import path from "path";
import { spawnSync } from "child_process";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const srcRoot = path.join(__dirname, "generated-src");
const packageRoot = path.join(__dirname, "generated-packages");

const longDescription = [
  "This package is intentionally generated for marketplace package-management tests.",
  "It exists to cover long-description display, search, pagination, and upload flows.",
  "The content is synthetic but packaged with Helm and valid for smoke-level upload checks.",
].join(" ");

const charts = [
  { name: "alpha-demo", version: "0.1.0", description: "Generated alpha demo chart." },
  { name: "beta-ops", version: "0.1.0", description: "Generated beta operations chart." },
  { name: "gamma-stack", version: "0.1.0", description: "Generated gamma stack chart." },
  { name: "delta-service", version: "0.1.0", description: "Generated delta service chart." },
  { name: "epsilon-tools", version: "0.1.0", description: "Generated epsilon tools chart." },
  { name: "zeta-observer", version: "0.1.0", description: "Generated zeta observer chart." },
  { name: "eta-runner", version: "0.1.0", description: "Generated eta runner chart." },
  { name: "theta-web", version: "0.1.0", description: "Generated theta web chart." },
  { name: "iota-api", version: "0.1.0", description: "Generated iota API chart." },
  { name: "kappa-worker", version: "0.1.0", description: "Generated kappa worker chart." },
  { name: "lambda-cache", version: "0.1.0", description: "Generated lambda cache chart." },
  { name: "mu-gateway", version: "0.1.0", description: "Generated mu gateway chart." },
  { name: "nu-scheduler", version: "0.1.0", description: "Generated nu scheduler chart." },
  { name: "pkg-2026-demo", version: "0.1.0", description: "Generated chart with digits in package name." },
  {
    name: "special-long-name-abcdefghijklmnopqrstuvwxyz0123456789abc",
    version: "0.1.0",
    description: `${longDescription} ${longDescription}`,
  },
];

const multiVersionBase = "multi-version-demo";
for (let i = 1; i <= 11; i += 1) {
  charts.push({
    name: multiVersionBase,
    version: `0.${i}.0`,
    description: `Generated multi-version chart release ${i}.`,
  });
}

function ensureDir(dir) {
  fs.mkdirSync(dir, { recursive: true });
}

function writeFile(filePath, content) {
  fs.writeFileSync(filePath, content, "utf8");
}

function buildChartSource(chart) {
  const chartDir = path.join(srcRoot, `${chart.name}-${chart.version}`);
  const templatesDir = path.join(chartDir, "templates");
  ensureDir(templatesDir);

  writeFile(
    path.join(chartDir, "Chart.yaml"),
    [
      "apiVersion: v2",
      `name: ${chart.name}`,
      `version: ${chart.version}`,
      'appVersion: "1.0.0"',
      `description: ${JSON.stringify(chart.description)}`,
      "type: application",
      "",
    ].join("\n"),
  );

  writeFile(
    path.join(chartDir, "values.yaml"),
    [
      "replicaCount: 1",
      "image:",
      "  repository: nginx",
      "  tag: stable",
      "  pullPolicy: IfNotPresent",
      "service:",
      "  type: ClusterIP",
      "  port: 80",
      "",
    ].join("\n"),
  );

  writeFile(
    path.join(templatesDir, "deployment.yaml"),
    [
      "apiVersion: apps/v1",
      "kind: Deployment",
      "metadata:",
      "  name: {{ .Chart.Name }}",
      "spec:",
      "  replicas: {{ .Values.replicaCount }}",
      "  selector:",
      "    matchLabels:",
      "      app: {{ .Chart.Name }}",
      "  template:",
      "    metadata:",
      "      labels:",
      "        app: {{ .Chart.Name }}",
      "    spec:",
      "      containers:",
      "        - name: nginx",
      '          image: "{{ .Values.image.repository }}:{{ .Values.image.tag }}"',
      "          imagePullPolicy: {{ .Values.image.pullPolicy }}",
      "          ports:",
      "            - containerPort: 80",
      "",
    ].join("\n"),
  );

  writeFile(
    path.join(templatesDir, "service.yaml"),
    [
      "apiVersion: v1",
      "kind: Service",
      "metadata:",
      "  name: {{ .Chart.Name }}",
      "spec:",
      "  type: {{ .Values.service.type }}",
      "  selector:",
      "    app: {{ .Chart.Name }}",
      "  ports:",
      "    - port: {{ .Values.service.port }}",
      "      targetPort: 80",
      "",
    ].join("\n"),
  );
}

function resolveHelmBinary() {
  const candidates = [
    "helm",
    path.join(process.env.LOCALAPPDATA || "", "Microsoft", "WinGet", "Links", "helm.exe"),
  ].filter(Boolean);

  for (const candidate of candidates) {
    const result = spawnSync(candidate, ["version"], { encoding: "utf8", shell: false });
    if (result.status === 0) {
      return candidate;
    }
  }

  throw new Error("Helm binary not found. Install Helm before running this generator.");
}

function packageChart(helmPath, chartDir) {
  const result = spawnSync(
    helmPath,
    ["package", chartDir, "--destination", packageRoot],
    { encoding: "utf8", shell: false },
  );
  if (result.status !== 0) {
    throw new Error(result.stderr || result.stdout || `Failed to package ${chartDir}`);
  }
}

function main() {
  fs.rmSync(srcRoot, { recursive: true, force: true });
  fs.rmSync(packageRoot, { recursive: true, force: true });
  ensureDir(srcRoot);
  ensureDir(packageRoot);

  charts.forEach(buildChartSource);

  const helmPath = resolveHelmBinary();
  charts.forEach((chart) => {
    const chartDir = path.join(srcRoot, `${chart.name}-${chart.version}`);
    packageChart(helmPath, chartDir);
  });

  const manifest = charts.map((chart) => ({
    name: chart.name,
    version: chart.version,
    packageFile: `${chart.name}-${chart.version}.tgz`,
    description: chart.description,
  }));

  writeFile(
    path.join(packageRoot, "manifest.json"),
    `${JSON.stringify({
      generatedAt: new Date().toISOString(),
      packageCount: manifest.length,
      distinctChartCount: new Set(manifest.map((item) => item.name)).size,
      notes: [
        "All generated packages are Helm application charts.",
        "Extension-component package format is not inferred in this generator.",
      ],
      packages: manifest,
    }, null, 2)}\n`,
  );

  console.log(`Generated ${manifest.length} chart packages in ${packageRoot}`);
}

main();
