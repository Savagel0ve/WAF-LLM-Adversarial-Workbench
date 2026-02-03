export enum AttackMethod {
  SAC = 'SAC (Baseline)',
  LLM_SEMANTIC = 'LLM Semantic Rewrite',
  LLM_RLHF = 'LLM RLHF-Tuned',
  LLM_AGENT = 'Multi-Agent CoT',
  LLM_EVOLUTIONARY = 'Evolutionary LLM (GA)',
  LLM_MCTS = 'MCTS Guided Search'
}

export enum WafProfile {
  MOD_SECURITY = 'ModSecurity (Regex-based)',
  CLOUDFLARE = 'Cloudflare (ML-based)',
  AWS_WAF = 'AWS WAF (Token-based)',
  CUSTOM_GATEWAY = 'Enterprise custom'
}

export enum AiModel {
  FLASH = 'gemini-3-flash-preview',
  PRO = 'gemini-3-pro-preview'
}

export enum WafStatus {
  BLOCKED = 'BLOCKED',
  BYPASSED = 'BYPASSED',
  PENDING = 'PENDING'
}

export interface PayloadLog {
  id: string;
  timestamp: string;
  original: string;
  mutated: string;
  method: AttackMethod;
  targetProfile: WafProfile;
  iteration: number;
  status: WafStatus;
  wafConfidence: number;
  reward: number;
  reasoning?: string;
  latency: number;
  modelUsed: AiModel;
}

export interface LiteratureRef {
  id: string;
  title: string;
  authors: string;
  year: number;
  contribution: string;
  gap: string;
}

export interface ThesisPoint {
  title: string;
  category: 'Algorithm' | 'Architecture' | 'Data' | 'Evaluation';
  description: string;
  innovation: string;
  difficulty: 'Medium' | 'High' | 'Very High';
}