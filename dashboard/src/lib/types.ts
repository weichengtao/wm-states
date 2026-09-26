export type Json =
  string | number | boolean | null | Json[] | { [key: string]: Json };
export type Settings = Record<string, Record<string, Json>>;
export type Field = {
  name: string;
  type: string;
  default: Json;
  choices?: string[];
  nullable?: boolean;
  description?: string;
};
export type Stage = {
  id: string;
  label: string;
  description: string;
  fields: Field[];
};
export type Schema = {
  stages: Stage[];
  presets: { example: Settings; smoke: Settings };
  defaults: Partial<RunRequest>;
};
export type RunRequest = {
  name: string;
  cache_dir: string;
  data_dir: string;
  stages: string[];
  n_jobs: number;
  session_list_file: string | null;
  max_sessions_to_run: number | null;
  figure_formats: string[];
  settings: Settings;
  allow_existing: boolean;
};
export type TemplateConfig = {
  settings: Settings;
  stages: string[];
  n_jobs: number;
  max_sessions_to_run: number | null;
  figure_formats: string[];
  data_dir?: string;
  session_list_file?: string | null;
};
export type PipelineTemplate = {
  id: string;
  name: string;
  description: string;
  builtin: boolean;
  created_at?: string;
  path?: string;
  config: TemplateConfig;
};
export type TemplateChange = {
  stage: string;
  field: string;
  label: string;
  before: Json | undefined;
  after: Json | undefined;
};
export type StageProgress = {
  stage: string;
  status: string;
  seconds?: number;
  error?: string;
};
export type Job = {
  id: string;
  name: string;
  cache_dir: string;
  status: string;
  created_at: string;
  started_at?: string;
  finished_at?: string;
  command: string;
  stages: StageProgress[];
  requested_stages: string[];
  logs: string[];
  exit_code?: number;
  error?: string;
};
export type Run = {
  id: string;
  name: string;
  path: string;
  updated_at: string;
  status: string;
  stages: StageProgress[];
  session_count: number;
  summary: Record<string, number | null>;
};
export type Session = {
  id: string;
  session: string;
  cue: number | null;
  trial_count: number;
  cell_count: number | null;
  selected_cells?: number | null;
  null_shuffles?: number | null;
  metrics: Record<string, number | null>;
  stages: string[];
};
export type SessionData = Session & {
  trial_ids: number[];
  time_bins: number[];
  observed: (number | null)[];
  null_mean: (number | null)[];
  null_low: (number | null)[];
  null_high: (number | null)[];
  accuracy: (number | null)[];
  total_off_durations: number[];
  max_off_durations: number[];
  errors: string[];
  warnings: string[];
};
export type Artifact = {
  id: string;
  path: string;
  name: string;
  stage: string;
  kind: string;
  session: string | null;
  url: string;
  bytes: number;
};
export type Manifest = {
  id: string;
  started_at?: string;
  finished_at?: string;
  status: string;
  stages: StageProgress[];
  settings?: Settings;
  invocation?: { command?: string; cwd?: string };
  [key: string]: unknown;
};
export type RunDetail = {
  run: Run;
  sessions: Session[];
  artifacts: Artifact[];
  manifests: Manifest[];
  errors?: string[];
};
export type TableData = {
  columns: string[];
  rows: Record<string, Json>[];
  total: number;
};
