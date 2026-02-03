import React from 'react';
import { LiteratureRef } from '../types';
import { BookOpen, ExternalLink, Hash, Link as LinkIcon, AlertCircle } from 'lucide-react';

const references: LiteratureRef[] = [
  {
    id: "REF-001",
    title: "XSS adversarial example attacks based on deep reinforcement learning",
    authors: "Chen et al.",
    year: 2022,
    contribution: "Proposed using DQN/SAC for discrete payload mutation.",
    gap: "Relies on fixed action sets; struggles with semantic logic complexity."
  },
  {
    id: "REF-002",
    title: "WAF-A-MoLE: Evading WAFs through Adversarial ML",
    authors: "Demetrio et al.",
    year: 2020,
    contribution: "Formalized the SQLi bypass as a search problem.",
    gap: "Search space exploration is slow; lacks the LLM's generative flexibility."
  },
  {
    id: "REF-003",
    title: "Large Language Models for Cybersecurity",
    authors: "Google DeepMind",
    year: 2024,
    contribution: "Showcased LLMs' ability to find 0-day logic flaws.",
    gap: "High token cost and lack of automated RL feedback loops for WAF specifically."
  }
];

export const LiteratureReview: React.FC = () => {
  return (
    <div className="space-y-8 max-w-5xl mx-auto pb-10">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-3xl font-bold text-white flex items-center gap-3">
            <BookOpen className="w-8 h-8 text-cyber-warning" />
            Literature Analysis & Positioning
          </h2>
          <p className="text-gray-400 mt-2">Mapping the gap between State-of-the-Art (SOTA) and our Thesis contributions.</p>
        </div>
      </div>

      <div className="grid grid-cols-1 gap-4">
        {references.map((ref) => (
          <div key={ref.id} className="bg-cyber-800 border border-cyber-700 rounded-xl p-6 hover:bg-cyber-800/80 transition-all border-l-4 border-l-cyber-warning">
            <div className="flex justify-between items-start mb-4">
              <div className="flex items-center gap-3">
                <div className="bg-cyber-900 px-2 py-1 rounded text-cyber-warning font-mono text-xs">{ref.id}</div>
                <h3 className="text-lg font-bold text-white">{ref.title}</h3>
              </div>
              <span className="text-gray-500 font-mono text-sm">{ref.authors}, {ref.year}</span>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="bg-cyber-900/40 p-4 rounded-lg border border-cyber-700/50">
                <div className="text-[10px] uppercase text-cyber-success font-bold mb-2 flex items-center gap-1">
                  <LinkIcon className="w-3 h-3" /> Base Contribution
                </div>
                <p className="text-sm text-gray-300">{ref.contribution}</p>
              </div>
              <div className="bg-cyber-danger/5 p-4 rounded-lg border border-cyber-danger/20">
                <div className="text-[10px] uppercase text-cyber-danger font-bold mb-2 flex items-center gap-1">
                  <AlertCircle className="w-3 h-3" /> Identified Research Gap
                </div>
                <p className="text-sm text-gray-400 italic">{ref.gap}</p>
              </div>
            </div>
          </div>
        ))}
      </div>

      <div className="bg-cyber-900/50 p-8 rounded-2xl border border-dashed border-cyber-700 text-center">
        <h4 className="text-xl font-bold text-cyber-accent mb-2">Our Thesis Proposition</h4>
        <p className="text-gray-400 max-w-2xl mx-auto leading-relaxed">
          Integrating <strong>LLM Reasoning</strong> as a High-Level Controller for <strong>Search-based Strategies</strong>, 
          bridging the gap between <u>fast but dumb</u> SAC mutations and <u>smart but expensive</u> LLM generations.
        </p>
      </div>
    </div>
  );
};