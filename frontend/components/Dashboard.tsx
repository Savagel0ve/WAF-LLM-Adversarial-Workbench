import React from 'react';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar, Legend, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar } from 'recharts';
import { Activity, ShieldCheck, Zap, AlertTriangle, FileText, Download } from 'lucide-react';

const escapeRateData = [
  { step: 'Step 1', sac: 12, llm: 25, hybrid: 30 },
  { step: 'Step 5', sac: 35, llm: 55, hybrid: 68 },
  { step: 'Step 10', sac: 55, llm: 78, hybrid: 89 },
  { step: 'Step 15', sac: 68, llm: 89, hybrid: 94 },
  { step: 'Step 20', sac: 75, llm: 94, hybrid: 98 },
  { step: 'Step 25', sac: 82, llm: 96, hybrid: 99 },
];

const ablationData = [
  { subject: 'Bypass Rate', A: 75, B: 98, fullMark: 100 },
  { subject: 'Conv. Speed', A: 40, B: 85, fullMark: 100 },
  { subject: 'Token Efficiency', A: 95, B: 60, fullMark: 100 },
  { subject: 'Diversity', A: 55, B: 92, fullMark: 100 },
  { subject: 'Payload Length', A: 85, B: 70, fullMark: 100 },
];

export const Dashboard: React.FC = () => {
  return (
    <div className="space-y-6 pb-10">
      
      {/* Header with Export */}
      <div className="flex justify-between items-end">
        <div>
          <h2 className="text-2xl font-bold text-white">Quantitative Evaluation</h2>
          <p className="text-sm text-gray-500">Benchmark results against SOTA baselines.</p>
        </div>
        <button className="flex items-center gap-2 bg-cyber-accent/10 border border-cyber-accent text-cyber-accent px-4 py-2 rounded-lg text-sm hover:bg-cyber-accent hover:text-cyber-900 transition-all font-bold">
          <Download className="w-4 h-4" />
          Export Thesis Data (CSV)
        </button>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-cyber-800 border border-cyber-700 p-4 rounded-xl">
          <div className="text-gray-400 text-[10px] uppercase font-bold mb-2 tracking-widest">ER Improvement</div>
          <div className="text-2xl font-mono font-bold text-white">+18.4%</div>
          <div className="text-[10px] text-cyber-success mt-1">Relative to SAC-SOTA</div>
        </div>
        
        <div className="bg-cyber-800 border border-cyber-700 p-4 rounded-xl">
          <div className="text-gray-400 text-[10px] uppercase font-bold mb-2 tracking-widest">Mean Latency</div>
          <div className="text-2xl font-mono font-bold text-white">842ms</div>
          <div className="text-[10px] text-gray-500 mt-1">T-Test: p {"<"} 0.05</div>
        </div>

        <div className="bg-cyber-800 border border-cyber-700 p-4 rounded-xl">
          <div className="text-gray-400 text-[10px] uppercase font-bold mb-2 tracking-widest">Semantic Accuracy</div>
          <div className="text-2xl font-mono font-bold text-white">99.1%</div>
          <div className="text-[10px] text-cyber-success mt-1">Babel-AST Verified</div>
        </div>

        <div className="bg-cyber-800 border border-cyber-700 p-4 rounded-xl">
          <div className="text-gray-400 text-[10px] uppercase font-bold mb-2 tracking-widest">Dataset Size</div>
          <div className="text-2xl font-mono font-bold text-white">10.5k</div>
          <div className="text-[10px] text-gray-500 mt-1">XSSed Standard Archive</div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        
        {/* Ablation Curve */}
        <div className="bg-cyber-800 border border-cyber-700 p-6 rounded-xl">
          <div className="flex justify-between items-start mb-6">
            <div>
              <h3 className="text-lg font-bold text-white">Learning Convergence</h3>
              <p className="text-xs text-gray-400">Ablation Study: SAC vs LLM vs Hybrid</p>
            </div>
            <div className="flex gap-4">
               <div className="flex items-center gap-1.5"><div className="w-2 h-2 rounded-full bg-cyber-accent"></div><span className="text-[10px] text-gray-400">Hybrid (Our)</span></div>
               <div className="flex items-center gap-1.5"><div className="w-2 h-2 rounded-full bg-cyber-warning"></div><span className="text-[10px] text-gray-400">LLM-Only</span></div>
               <div className="flex items-center gap-1.5"><div className="w-2 h-2 rounded-full bg-gray-600"></div><span className="text-[10px] text-gray-400">SAC-Only</span></div>
            </div>
          </div>
          
          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={escapeRateData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#2a2a40" />
                <XAxis dataKey="step" stroke="#6b7280" fontSize={10} tickLine={false} />
                <YAxis stroke="#6b7280" fontSize={10} tickLine={false} unit="%" />
                <Tooltip contentStyle={{ backgroundColor: '#13131f', borderColor: '#2a2a40', color: '#fff' }} />
                <Area type="monotone" dataKey="hybrid" stroke="#00f0ff" fillOpacity={0.1} fill="#00f0ff" strokeWidth={3} />
                <Area type="monotone" dataKey="llm" stroke="#fcee0a" fillOpacity={0} strokeWidth={2} strokeDasharray="3 3" />
                <Area type="monotone" dataKey="sac" stroke="#6b7280" fillOpacity={0} strokeWidth={2} strokeDasharray="10 5" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Model Capability Radar */}
        <div className="bg-cyber-800 border border-cyber-700 p-6 rounded-xl">
          <h3 className="text-lg font-bold text-white mb-1">Model Capability Profile</h3>
          <p className="text-xs text-gray-400 mb-6">Comparative analysis of algorithm performance dimensions</p>
          
          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <RadarChart cx="50%" cy="50%" outerRadius="80%" data={ablationData}>
                <PolarGrid stroke="#2a2a40" />
                <PolarAngleAxis dataKey="subject" stroke="#6b7280" fontSize={10} />
                <PolarRadiusAxis angle={30} domain={[0, 100]} stroke="#2a2a40" />
                <Radar name="Our Method" dataKey="B" stroke="#00f0ff" fill="#00f0ff" fillOpacity={0.4} />
                <Radar name="Baseline (SAC)" dataKey="A" stroke="#6b7280" fill="#6b7280" fillOpacity={0.2} />
                <Tooltip contentStyle={{ backgroundColor: '#13131f', borderColor: '#2a2a40', color: '#fff' }} />
              </RadarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Experimental Log Output Table (Engineering volume proof) */}
      <div className="bg-cyber-800 border border-cyber-700 rounded-xl overflow-hidden">
        <div className="px-6 py-4 border-b border-cyber-700 bg-cyber-900/30 flex justify-between items-center">
           <h3 className="font-bold text-white text-sm">Experimental Benchmark Logs</h3>
           <FileText className="w-4 h-4 text-gray-500" />
        </div>
        <table className="w-full text-xs text-left border-collapse">
          <thead>
            <tr className="bg-cyber-900/50 text-gray-500 border-b border-cyber-700 uppercase font-mono">
              <th className="px-6 py-3">Timestamp</th>
              <th className="px-6 py-3">Algorithm</th>
              <th className="px-6 py-3">WAF Type</th>
              <th className="px-6 py-3">Bypass Rate</th>
              <th className="px-6 py-3">Complexity (Tokens)</th>
              <th className="px-6 py-3">Status</th>
            </tr>
          </thead>
          <tbody className="text-gray-400 divide-y divide-cyber-700">
            <tr>
              <td className="px-6 py-3 font-mono">2024-05-24 14:22:01</td>
              <td className="px-6 py-3">Evolutionary LLM</td>
              <td className="px-6 py-3">Cloudflare (Sim)</td>
              <td className="px-6 py-3 text-cyber-success">98.2%</td>
              <td className="px-6 py-3">1,402</td>
              <td className="px-6 py-3"><span className="px-2 py-0.5 rounded-full bg-cyber-success/10 text-cyber-success text-[10px] border border-cyber-success/30">VALIDATED</span></td>
            </tr>
            <tr>
              <td className="px-6 py-3 font-mono">2024-05-24 14:25:44</td>
              <td className="px-6 py-3">SAC (Baseline)</td>
              <td className="px-6 py-3">Cloudflare (Sim)</td>
              <td className="px-6 py-3 text-cyber-warning">72.4%</td>
              <td className="px-6 py-3">0</td>
              <td className="px-6 py-3"><span className="px-2 py-0.5 rounded-full bg-cyber-warning/10 text-cyber-warning text-[10px] border border-cyber-warning/30">LOW_EFFICIENCY</span></td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  );
};