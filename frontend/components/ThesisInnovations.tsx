
import React, { useState } from 'react';
import { ThesisPoint } from '../types';
import { BrainCircuit, Network, Zap, Target, ArrowRight, Microscope, GitMerge, SearchCode, Database, Code2, ChevronDown, ChevronUp } from 'lucide-react';

const innovations: ThesisPoint[] = [
  {
    title: "LLM-as-Mutation-Operator (GA)",
    category: "Algorithm",
    description: "Combine Genetic Algorithms with LLMs where the model acts as a smart 'crossover' and 'mutation' operator.",
    innovation: "Instead of random bit-flips, the LLM uses high-level reasoning to 'breed' two successful payloads or mutate a payload based on the 'fitness' (WAF bypass score).",
    difficulty: "High"
  },
  {
    title: "MCTS Guided Payload Space Search",
    category: "Algorithm",
    description: "Use Monte Carlo Tree Search (MCTS) to explore the vast combinatorial space of bypasses.",
    innovation: "Treating bypass as a game. The LLM predicts possible mutation branches, and MCTS evaluates the 'win rate' (bypass probability), focusing compute on the most promising paths.",
    difficulty: "Very High"
  },
  {
    title: "Contrastive WAF Boundary Learning",
    category: "Data",
    description: "Train a small student model to map the WAF's decision boundary using LLM-generated positive/negative pairs.",
    innovation: "By generating hundreds of 'near-miss' payloads (almost blocked vs just passed), the attacker learns the exact regex or token limits of the target WAF without a local oracle.",
    difficulty: "Medium"
  },
  {
    title: "Multi-Agent Adversarial Gaming",
    category: "Architecture",
    description: "Generator Agent (Attacker) vs Discriminator Agent (Defender) zero-sum game with Chain of Thought.",
    innovation: "The Generator explains its logic, and the Discriminator provides counter-reasons. This mimics a GAN architecture but leverages the textual reasoning of LLMs.",
    difficulty: "High"
  }
];

export const ThesisInnovations: React.FC = () => {
  const [expandedAlgo, setExpandedAlgo] = useState<string | null>('sac');

  return (
    <div className="space-y-8 max-w-5xl mx-auto pb-10">
      
      <div className="space-y-4">
        <h2 className="text-3xl font-bold text-white flex items-center gap-3">
          <Microscope className="w-8 h-8 text-cyber-accent" />
          Research Methodology Explorer
        </h2>
        <p className="text-gray-400 max-w-3xl leading-relaxed">
          While <span className="text-cyber-accent font-mono">SAC (Soft Actor-Critic)</span> is a strong baseline, 
          recent literature suggests that combining LLMs with <strong>Search-based Software Engineering (SBSE)</strong> 
          offers superior convergence.
        </p>
      </div>

      {/* Algorithm Deep Dive Section */}
      <div className="bg-cyber-800 border border-cyber-700 rounded-2xl overflow-hidden shadow-xl">
        <div 
          className="p-4 bg-cyber-700/30 flex justify-between items-center cursor-pointer hover:bg-cyber-700/50 transition-colors"
          onClick={() => setExpandedAlgo(expandedAlgo === 'sac' ? null : 'sac')}
        >
          <div className="flex items-center gap-3">
            <Database className="w-5 h-5 text-cyber-accent" />
            <h3 className="font-bold text-white">Technical Implementation: SAC Baseline</h3>
          </div>
          {expandedAlgo === 'sac' ? <ChevronUp className="w-5 h-5 text-gray-500" /> : <ChevronDown className="w-5 h-5 text-gray-500" />}
        </div>
        
        {expandedAlgo === 'sac' && (
          <div className="p-6 grid grid-cols-1 md:grid-cols-3 gap-8 animate-in fade-in duration-300">
            <div className="space-y-4">
              <div className="text-xs font-bold text-cyber-accent uppercase tracking-widest flex items-center gap-2">
                <div className="w-1.5 h-1.5 rounded-full bg-cyber-accent"></div>
                State & Action
              </div>
              <ul className="space-y-3">
                <li className="bg-cyber-900/50 p-3 rounded-lg border border-cyber-700">
                  <span className="text-white text-sm block font-bold mb-1">State ($S$)</span>
                  <p className="text-[11px] text-gray-400">Vectorized representation of the Payload (e.g., Word2Vec or Char-level LSTM Embedding).</p>
                </li>
                <li className="bg-cyber-900/50 p-3 rounded-lg border border-cyber-700">
                  <span className="text-white text-sm block font-bold mb-1">Action ($A$)</span>
                  <p className="text-[11px] text-gray-400">Discrete set of 27+ mutations (Encoding, Comment Injection, Case Swapping).</p>
                </li>
              </ul>
            </div>

            <div className="space-y-4">
              <div className="text-xs font-bold text-cyber-success uppercase tracking-widest flex items-center gap-2">
                <div className="w-1.5 h-1.5 rounded-full bg-cyber-success"></div>
                Optimization
              </div>
              <ul className="space-y-3">
                <li className="bg-cyber-900/50 p-3 rounded-lg border border-cyber-700">
                  <span className="text-white text-sm block font-bold mb-1">Entropy Max.</span>
                  {/* FIX: Escaped curly braces in LaTeX formula to prevent them from being interpreted as JSX expressions */}
                  <p className="text-[11px] text-gray-400">$\pi^* = \arg\max \mathbb&#123;E&#125;[\sum (R_t + \alpha \mathcal&#123;H&#125;(\pi))]$ - Prevents collapsing to easy regex patterns.</p>
                </li>
                <li className="bg-cyber-900/50 p-3 rounded-lg border border-cyber-700">
                  <span className="text-white text-sm block font-bold mb-1">Twin Q-Learning</span>
                  <p className="text-[11px] text-gray-400">Two Critic networks estimate $Q(s,a)$ to prevent overestimation bias in black-box WAF targets.</p>
                </li>
              </ul>
            </div>

            <div className="space-y-4">
              <div className="text-xs font-bold text-cyber-warning uppercase tracking-widest flex items-center gap-2">
                <div className="w-1.5 h-1.5 rounded-full bg-cyber-warning"></div>
                Reward Design
              </div>
              <div className="bg-cyber-900/50 p-4 rounded-xl border border-cyber-warning/20">
                <div className="text-[10px] font-mono text-gray-500 mb-2">PSEUDOCODE</div>
                <pre className="text-[10px] text-cyber-warning leading-tight font-mono">
{`if status == "Bypassed":
  reward = 10.0 + (1 / length)
elif status == "Blocked":
  reward = log(1 - confidence)
else:
  reward = -0.1 (time penalty)`}
                </pre>
              </div>
            </div>
          </div>
        )}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {innovations.map((point, index) => (
          <div key={index} className="bg-cyber-800/50 border border-cyber-700 rounded-2xl p-6 hover:border-cyber-accent/50 transition-all hover:translate-y-[-4px] group">
            <div className="flex justify-between items-start mb-4">
              <div className="p-3 bg-cyber-700 rounded-lg group-hover:bg-cyber-accent/10 transition-colors">
                {point.title.includes('GA') && <GitMerge className="w-6 h-6 text-cyber-success" />}
                {point.title.includes('MCTS') && <SearchCode className="w-6 h-6 text-cyber-warning" />}
                {point.category === 'Architecture' && <Network className="w-6 h-6 text-cyber-accent" />}
                {point.category === 'Data' && <Target className="w-6 h-6 text-cyber-danger" />}
              </div>
              <span className={`px-3 py-1 rounded-full text-[10px] font-bold border ${
                point.difficulty === 'Very High' ? 'border-purple-500/30 text-purple-400 bg-purple-500/5' :
                point.difficulty === 'High' ? 'border-cyber-danger/30 text-cyber-danger bg-cyber-danger/5' :
                'border-cyber-warning/30 text-cyber-warning bg-cyber-warning/5'
              }`}>
                {point.difficulty} Effort
              </span>
            </div>
            
            <h3 className="text-xl font-bold text-white mb-2">{point.title}</h3>
            <p className="text-sm text-gray-400 mb-4 h-16">{point.description}</p>
            
            <div className="bg-cyber-900/50 p-4 rounded-xl border border-cyber-700/50">
              <div className="flex items-center gap-2 mb-2 text-cyber-success text-xs font-bold uppercase tracking-wider">
                <Zap className="w-3 h-3" />
                Key Innovation
              </div>
              <p className="text-sm text-gray-300 leading-snug">
                {point.innovation}
              </p>
            </div>
          </div>
        ))}
      </div>

      <div className="bg-gradient-to-r from-cyber-800 to-cyber-900 rounded-2xl p-8 border border-cyber-700 relative overflow-hidden">
        <div className="absolute top-0 right-0 w-64 h-64 bg-cyber-accent/5 rounded-full blur-3xl -translate-y-1/2 translate-x-1/2"></div>
        
        <h3 className="text-2xl font-bold text-white mb-6 relative z-10">Advanced Pipeline: Evolutionary LLM Search</h3>
        
        <div className="flex flex-col md:flex-row items-center gap-4 relative z-10">
          
          <div className="flex-1 bg-cyber-900 border border-cyber-600 p-4 rounded-xl text-center group cursor-help">
            <div className="font-mono text-cyber-accent mb-2 font-bold">Population P</div>
            <div className="text-xs text-gray-400">N candidate payloads</div>
            <div className="mt-2 text-[10px] bg-cyber-800 py-1 rounded">Diversity: High</div>
          </div>

          <ArrowRight className="text-gray-600 w-6 h-6 hidden md:block" />
          
          <div className="flex-1 bg-cyber-900 border border-cyber-success/50 p-4 rounded-xl text-center shadow-[0_0_15px_rgba(5,255,161,0.1)]">
            <div className="font-mono text-cyber-success mb-2 font-bold">LLM Mutator</div>
            <div className="text-xs text-gray-400">GPT-4 / Gemini 2.5</div>
            <div className="mt-2 text-[10px] bg-cyber-800 py-1 rounded">"Cross-breed these bips"</div>
          </div>

          <ArrowRight className="text-gray-600 w-6 h-6 hidden md:block" />

          <div className="flex-1 bg-cyber-900 border border-cyber-warning/50 p-4 rounded-xl text-center">
            <div className="font-mono text-cyber-warning mb-2 font-bold">Fitness (WAF)</div>
            <div className="text-xs text-gray-400">Target Environment</div>
            <div className="mt-2 text-[10px] bg-cyber-800 py-1 rounded">Score = 1/Confidence</div>
          </div>

        </div>

        <div className="mt-8 grid grid-cols-1 md:grid-cols-3 gap-4">
           <div className="p-3 bg-cyber-900/80 rounded-lg border border-cyber-700 text-xs text-gray-400">
             <span className="text-cyber-accent font-bold block mb-1">vs SAC</span>
             Handles discrete action spaces and logic jumps better than gradients.
           </div>
           <div className="p-3 bg-cyber-900/80 rounded-lg border border-cyber-700 text-xs text-gray-400">
             <span className="text-cyber-accent font-bold block mb-1">Complexity</span>
             Requires population management and token budget optimization.
           </div>
           <div className="p-3 bg-cyber-900/80 rounded-lg border border-cyber-700 text-xs text-gray-400">
             <span className="text-cyber-accent font-bold block mb-1">State of Art</span>
             Currently unexplored in WAF context (Blue Ocean Research).
           </div>
        </div>
      </div>

    </div>
  );
};
