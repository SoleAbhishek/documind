import React, { useState, useEffect, useCallback, useMemo } from 'react';
import Navbar from './components/Navbar';
import DocumentSidebar from './components/DocumentSidebar';
import ChatArea from './components/ChatArea';
import ProvenanceInspector from './components/ProvenanceInspector';
import {
  fetchHealth,
  fetchTenants,
  createTenant,
  fetchDocuments,
  uploadDocument,
  deleteDocument,
  fetchSessions,
  fetchSessionMessages,
  streamChatQuery
} from './api/client';

export default function App() {
  const [health, setHealth] = useState(null);
  const [tenants, setTenants] = useState([]);
  const [activeTenantId, setActiveTenantId] = useState(null);

  const [documents, setDocuments] = useState([]);
  const [selectedDocFilter, setSelectedDocFilter] = useState(null);
  const [isUploading, setIsUploading] = useState(false);

  const [sessions, setSessions] = useState([]);
  const [activeSessionId, setActiveSessionId] = useState(null);
  const [messages, setMessages] = useState([]);

  // Citations & Provenance
  const [citations, setCitations] = useState([]);
  const [selectedSourceId, setSelectedSourceId] = useState(null);

  // Streaming State
  const [isGenerating, setIsGenerating] = useState(false);
  const [streamingMessage, setStreamingMessage] = useState('');
  const [currentStage, setCurrentStage] = useState('');

  // 1. Initial Load: Health and Tenants
  useEffect(() => {
    async function init() {
      try {
        const [healthData, tenantsData] = await Promise.all([
          fetchHealth().catch(() => null),
          fetchTenants().catch(() => [])
        ]);
        setHealth(healthData);
        setTenants(tenantsData);

        if (tenantsData.length > 0) {
          setActiveTenantId(tenantsData[0].id);
        }
      } catch (err) {
        console.error('Initialization error:', err);
      }
    }
    init();
  }, []);

  // 2. Load Documents & Sessions when Active Tenant changes
  const loadTenantData = useCallback(async (tenantId) => {
    if (!tenantId) return;
    try {
      const [docsData, sessionsData] = await Promise.all([
        fetchDocuments(tenantId).catch(() => []),
        fetchSessions(tenantId).catch(() => [])
      ]);
      setDocuments(docsData);
      setSessions(sessionsData);
    } catch (err) {
      console.error('Failed to load tenant data:', err);
    }
  }, []);

  useEffect(() => {
    if (activeTenantId) {
      loadTenantData(activeTenantId);
      // Reset chat on tenant switch
      setActiveSessionId(null);
      setMessages([]);
      setCitations([]);
      setSelectedSourceId(null);
      setSelectedDocFilter(null);
    }
  }, [activeTenantId, loadTenantData]);

  // 3. Document Polling: auto-refresh while any document is PENDING or PARSING
  useEffect(() => {
    const hasUnfinished = documents.some(
      (d) => d.status === 'PENDING' || d.status === 'PARSING'
    );
    if (!hasUnfinished || !activeTenantId) return;

    const timer = setInterval(async () => {
      try {
        const updated = await fetchDocuments(activeTenantId);
        setDocuments(updated);
      } catch (e) {
        console.error('Polling error:', e);
      }
    }, 2500);

    return () => clearInterval(timer);
  }, [documents, activeTenantId]);

  // Lookup map for documents by id
  const documentsMap = useMemo(() => {
    const map = {};
    for (const d of documents) {
      map[d.id] = d;
    }
    return map;
  }, [documents]);

  // Handle Tenant Creation
  const handleCreateTenant = async (name) => {
    const newTenant = await createTenant(name);
    setTenants((prev) => [newTenant, ...prev]);
    setActiveTenantId(newTenant.id);
  };

  // Handle Document Upload
  const handleUploadFile = async (file) => {
    if (!activeTenantId) {
      alert('Please select or create a tenant first.');
      return;
    }
    setIsUploading(true);
    try {
      await uploadDocument(activeTenantId, file);
      // Refresh documents list immediately
      const updatedDocs = await fetchDocuments(activeTenantId);
      setDocuments(updatedDocs);
    } catch (err) {
      alert(`Upload error: ${err.message}`);
    } finally {
      setIsUploading(false);
    }
  };

  // Handle Document Deletion
  const handleDeleteDocument = async (docId) => {
    try {
      await deleteDocument(activeTenantId, docId);
      setDocuments((prev) => prev.filter((d) => d.id !== docId));
      if (selectedDocFilter === docId) {
        setSelectedDocFilter(null);
      }
    } catch (err) {
      alert(`Delete error: ${err.message}`);
    }
  };

  // Handle Select Citation
  const handleSelectCitation = (sourceId) => {
    setSelectedSourceId(sourceId);
    // Smooth scroll in inspector drawer
    const el = document.getElementById(`citation-${sourceId}`);
    if (el) {
      el.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
  };

  // Handle New Chat
  const handleNewChat = () => {
    setActiveSessionId(null);
    setMessages([]);
    setCitations([]);
    setSelectedSourceId(null);
    setStreamingMessage('');
    setCurrentStage('');
  };

  // Handle Send Message (with SSE streaming)
  const handleSendMessage = async (queryText) => {
    if (!activeTenantId || !queryText.trim() || isGenerating) return;

    // Add user message to UI immediately
    const userMessage = {
      id: `temp-${Date.now()}`,
      role: 'user',
      content: queryText
    };
    setMessages((prev) => [...prev, userMessage]);

    setIsGenerating(true);
    setStreamingMessage('');
    setCurrentStage('Retrieving via Qdrant (Dense + BM25 RRF)...');

    let accumulatedTokens = '';
    let currentCitations = [];
    let assignedSessionId = activeSessionId;

    await streamChatQuery({
      tenantId: activeTenantId,
      query: queryText,
      sessionId: activeSessionId,
      documentId: selectedDocFilter,
      onEvent: (event) => {
        if (event.type === 'session') {
          assignedSessionId = event.session_id;
          setActiveSessionId(event.session_id);
        } else if (event.type === 'status') {
          setCurrentStage(event.message || event.stage);
        } else if (event.type === 'citations') {
          currentCitations = event.citations || [];
          setCitations(currentCitations);
          if (currentCitations.length > 0) {
            setSelectedSourceId(currentCitations[0].source_id);
          }
        } else if (event.type === 'token') {
          accumulatedTokens += event.content;
          setStreamingMessage(accumulatedTokens);
        }
      },
      onError: (err) => {
        console.error('Streaming error:', err);
        const errorMsg = {
          id: `err-${Date.now()}`,
          role: 'assistant',
          content: `An error occurred: ${err.message}`
        };
        setMessages((prev) => [...prev, errorMsg]);
        setIsGenerating(false);
        setStreamingMessage('');
        setCurrentStage('');
      },
      onDone: (doneEvent) => {
        const assistantMessage = {
          id: doneEvent.message_id || `asst-${Date.now()}`,
          role: 'assistant',
          content: accumulatedTokens,
          retrieved_chunks: doneEvent.citations || currentCitations
        };
        setMessages((prev) => [...prev, assistantMessage]);
        setIsGenerating(false);
        setStreamingMessage('');
        setCurrentStage('');
      }
    });
  };

  return (
    <div className="flex flex-col h-screen w-screen overflow-hidden bg-slate-950 text-slate-100">
      {/* Top Navbar */}
      <Navbar
        tenants={tenants}
        activeTenantId={activeTenantId}
        onSelectTenant={setActiveTenantId}
        onCreateTenant={handleCreateTenant}
        health={health}
      />

      {/* Main 3-Column Split-Screen Layout */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left: Document Knowledge Base Sidebar */}
        <DocumentSidebar
          documents={documents}
          isUploading={isUploading}
          onUploadFile={handleUploadFile}
          onDeleteDocument={handleDeleteDocument}
          selectedDocId={selectedDocFilter}
          onSelectDocFilter={setSelectedDocFilter}
        />

        {/* Center: Interactive RAG Chat Area */}
        <ChatArea
          messages={messages}
          streamingMessage={streamingMessage}
          currentStage={currentStage}
          onSendMessage={handleSendMessage}
          onNewChat={handleNewChat}
          onSelectCitation={handleSelectCitation}
          selectedSourceId={selectedSourceId}
          isGenerating={isGenerating}
        />

        {/* Right: Citation Provenance & Confidence Inspector */}
        <ProvenanceInspector
          citations={citations}
          selectedSourceId={selectedSourceId}
          onSelectCitation={handleSelectCitation}
          documentsMap={documentsMap}
        />
      </div>
    </div>
  );
}
