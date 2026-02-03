import React from 'react';
import { LayoutDashboard, Lightbulb, Terminal, BookOpen, Microscope, GraduationCap } from 'lucide-react';
import { TabType } from '../App';

interface SidebarProps {
  activeTab: TabType;
  setActiveTab: (tab: TabType) => void;
}

export const Sidebar: React.FC<SidebarProps> = ({ activeTab, setActiveTab }) => {
  const navItems = [
    { id: 'innovations', label: 'Innovation Map', icon: Lightbulb, desc: 'Research Scope' },
    { id: 'literature', label: 'Literature Review', icon: GraduationCap, desc: 'Academic Gap' },
    { id: 'dashboard', label: 'Ablation Analysis', icon: LayoutDashboard, desc: 'Comparative Data' },
    { id: 'console', label: 'Workbench', icon: Terminal, desc: 'Attack Simulation' },
  ];

  return (
    <aside className="w-72 border-r border-cyber-700 bg-cyber-800 flex flex-col">
      <div className="p-6">
        <div className="text-xs font-bold text-cyber-500 uppercase tracking-widest mb-4">Academic Module</div>
        <nav className="space-y-2">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = activeTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => setActiveTab(item.id as any)}
                className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl transition-all duration-200 group ${
                  isActive 
                    ? 'bg-cyber-700/50 border border-cyber-600 text-cyber-accent shadow-[0_0_15px_rgba(0,240,255,0.1)]' 
                    : 'text-gray-400 hover:bg-cyber-700/30 hover:text-white'
                }`}
              >
                <Icon className={`w-5 h-5 ${isActive ? 'text-cyber-accent' : 'text-gray-500 group-hover:text-white'}`} />
                <div className="text-left">
                  <div className="font-medium text-sm">{item.label}</div>
                  <div className="text-[10px] opacity-60 font-mono uppercase">{item.desc}</div>
                </div>
              </button>
            );
          })}
        </nav>
      </div>

      <div className="mt-auto p-6 border-t border-cyber-700">
        <div className="p-4 rounded-xl bg-gradient-to-br from-cyber-700 to-cyber-800 border border-cyber-600">
          <div className="flex items-center gap-2 mb-2">
            <Microscope className="w-4 h-4 text-cyber-accent" />
            <span className="text-xs font-bold text-white">Thesis Status</span>
          </div>
          <div className="space-y-1.5">
            <div className="flex justify-between text-[10px]">
              <span className="text-gray-500">Code Coverage</span>
              <span className="text-cyber-success">92%</span>
            </div>
            <div className="w-full bg-cyber-900 h-1 rounded-full overflow-hidden">
              <div className="bg-cyber-success h-full w-[92%]"></div>
            </div>
            <div className="flex justify-between text-[10px]">
              <span className="text-gray-500">Exp. Validation</span>
              <span className="text-cyber-warning">75%</span>
            </div>
            <div className="w-full bg-cyber-900 h-1 rounded-full overflow-hidden">
              <div className="bg-cyber-warning h-full w-[75%]"></div>
            </div>
          </div>
        </div>
      </div>
    </aside>
  );
};