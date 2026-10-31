import React, { useState } from 'react';
import { Database, Plus, Sparkles, Building2, Layers, CheckCircle2 } from 'lucide-react';

export default function Navbar({
  tenants,
  activeTenantId,
  onSelectTenant,
  onCreateTenant,
  health
}) {
  const [showModal, setShowModal] = useState(false);
  const [newTenantName, setNewTenantName] = useState('');
  const [isCreating, setIsCreating] = useState(false);

  const handleCreate = async (e) => {
    e.preventDefault();
    if (!newTenantName.trim()) return;
    setIsCreating(true);
    try {
      await onCreateTenant(newTenantName.trim());
      setNewTenantName('');
      setShowModal(false);
    } catch (err) {
      alert(`Error creating tenant: ${err.message}`);
    } finally {
      setIsCreating(false);
    }
  };

  return (
    <>
      <header className="h-16 border-b border-slate-800 bg-slate-900/90 backdrop-blur px-6 flex items-center justify-between z-20">
        {/* Brand */}
        <div className="flex items-center space-x-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-indigo-600 via-indigo-500 to-cyan-400 flex items-center justify-center shadow-lg shadow-indigo-500/20">
            <Sparkles className="w-5 h-5 text-white" />
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <span className="font-bold text-lg tracking-tight bg-gradient-to-r from-white via-slate-200 to-slate-400 bg-clip-text text-transparent">
                DocuMind
              </span>
              <span className="text-[10px] font-semibold px-2 py-0.5 rounded-full bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">
                Hybrid RAG
              </span>
            </div>
            <p className="text-xs text-slate-400">Enterprise Multi-Tenant Intelligence</p>
          </div>
        </div>

        {/* Tenant Switcher & Engine Pills */}
        <div className="flex items-center space-x-4">
          {/* Active Model Badges */}
          <div className="hidden lg:flex items-center space-x-2 text-xs">
            <div className="px-2.5 py-1 rounded-md bg-slate-800/80 border border-slate-700/60 text-slate-300 flex items-center space-x-1.5">
              <Layers className="w-3.5 h-3.5 text-cyan-400" />
              <span>Dense 768 + BM25 RRF</span>
            </div>
            <div className="px-2.5 py-1 rounded-md bg-slate-800/80 border border-slate-700/60 text-slate-300 flex items-center space-x-1.5">
              <Sparkles className="w-3.5 h-3.5 text-amber-400" />
              <span>Gemini 3.1 Flash Lite</span>
            </div>
          </div>

          {/* Tenant Selector Dropdown */}
          <div className="flex items-center bg-slate-800/90 rounded-lg p-1 border border-slate-700">
            <div className="px-2 flex items-center text-slate-400">
              <Building2 className="w-4 h-4 text-indigo-400 mr-1.5" />
              <span className="text-xs font-medium">Tenant:</span>
            </div>
            <select
              value={activeTenantId || ''}
              onChange={(e) => onSelectTenant(e.target.value)}
              className="bg-transparent text-xs font-semibold text-white px-2 py-1 outline-none cursor-pointer hover:text-indigo-300 transition-colors"
            >
              {tenants.map((t) => (
                <option key={t.id} value={t.id} className="bg-slate-900 text-white">
                  {t.name}
                </option>
              ))}
            </select>
            <button
              onClick={() => setShowModal(true)}
              className="ml-1 p-1 rounded-md hover:bg-slate-700 text-slate-400 hover:text-white transition-colors"
              title="Add New Tenant"
            >
              <Plus className="w-4 h-4" />
            </button>
          </div>

          {/* Health Indicator */}
          <div className="flex items-center space-x-1.5 text-xs text-emerald-400 bg-emerald-500/10 border border-emerald-500/20 px-2.5 py-1 rounded-md">
            <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
            <span className="font-medium hidden sm:inline">Online</span>
          </div>
        </div>
      </header>

      {/* Create Tenant Modal */}
      {showModal && (
        <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50 p-4">
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 max-w-sm w-full shadow-2xl">
            <h3 className="text-lg font-bold text-white mb-2">Create New Tenant</h3>
            <p className="text-xs text-slate-400 mb-4">
              Tenants get completely isolated PostgreSQL records, Qdrant vectors, and MinIO storage paths.
            </p>
            <form onSubmit={handleCreate}>
              <input
                type="text"
                placeholder="e.g. Acme Health Corp"
                value={newTenantName}
                onChange={(e) => setNewTenantName(e.target.value)}
                autoFocus
                className="w-full px-3 py-2 bg-slate-950 border border-slate-800 rounded-lg text-sm text-white placeholder-slate-500 focus:outline-none focus:border-indigo-500 mb-4"
              />
              <div className="flex justify-end space-x-2">
                <button
                  type="button"
                  onClick={() => setShowModal(false)}
                  className="px-3 py-1.5 text-xs text-slate-400 hover:text-white rounded-lg hover:bg-slate-800"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={isCreating || !newTenantName.trim()}
                  className="px-4 py-1.5 text-xs font-semibold bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg transition-colors disabled:opacity-50"
                >
                  {isCreating ? 'Creating...' : 'Create Tenant'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </>
  );
}
