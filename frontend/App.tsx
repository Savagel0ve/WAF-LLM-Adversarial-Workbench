import React, { useState } from 'react';
import { Sidebar } from './components/Sidebar';
import { ThesisInnovations } from './components/ThesisInnovations';
import { AttackConsole } from './components/AttackConsole';
import { Dashboard } from './components/Dashboard';
import { LiteratureReview } from './components/LiteratureReview';
import { ShieldAlert } from 'lucide-react';

export type TabType = 'dashboard' | 'innovations' | 'console' | 'literature';

export default function App() {
  const [activeTab, setActiveTab] = useState<TabType>('innovations');

  return (
    <div className="flex h-screen bg-cyber-900 text-gray-200 font-sans overflow-hidden">
      <Sidebar activeTab={activeTab} setActiveTab={setActiveTab} />

      <main className="flex-1 flex flex-col min-w-0 overflow-hidden">
        <header className="h-16 border-b border-cyber-700 bg-cyber-800/50 backdrop-blur flex items-center px-6 justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-cyber-accent/10 rounded-lg">
              <ShieldAlert className="w-6 h-6 text-cyber-accent" />
            </div>
            <div>
              <h1 className="text-lg font-bold text-white tracking-wide uppercase">WAF-LLM Adversarial Research System</h1>
              <p className="text-xs text-gray-500 font-mono">Experimental Framework for Graduate Thesis</p>
            </div>
          </div>
          <div className="flex items-center gap-4">
             <div className="px-3 py-1 rounded-full bg-cyber-700 border border-cyber-600 text-[10px] font-mono text-cyber-accent">
                NODE_STABLE :: 10.2.45
             </div>
             <div className="flex items-center gap-2">
                <span className="text-[10px] text-gray-500 font-mono">ORACLE_LINK</span>
                <div className="w-2 h-2 rounded-full bg-cyber-success animate-pulse shadow-[0_0_10px_#05ffa1]"></div>
             </div>
          </div>
        </header>

        <div className="flex-1 overflow-y-auto p-6 scroll-smooth">
          {activeTab === 'innovations' && <ThesisInnovations />}
          {activeTab === 'dashboard' && <Dashboard />}
          {activeTab === 'console' && <AttackConsole />}
          {activeTab === 'literature' && <LiteratureReview />}
        </div>
      </main>
    </div>
  );
}