import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { Toaster } from "@/components/ui/toaster";
import { TooltipProvider } from "@/components/ui/tooltip";
import Layout from "@/components/Layout";
import { AnalysisProvider } from "@/hooks/useAnalysis";
import Dashboard from "@/pages/Dashboard";
import DataInput from "@/pages/DataInput";
import Preprocessing from "@/pages/Preprocessing";
import TopicModeling from "@/pages/TopicModeling";
import Sentiment from "@/pages/Sentiment";
import Visualizations from "@/pages/Visualizations";
import Summarization from "@/pages/Summarization";
import History from "@/pages/History";
import NotFound from "@/pages/NotFound";

const queryClient = new QueryClient();

const App = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <Toaster />
      <Sonner />
      <AnalysisProvider>
        <BrowserRouter>
          <Layout>
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/input" element={<DataInput />} />
              <Route path="/preprocessing" element={<Preprocessing />} />
              <Route path="/topic-modeling" element={<TopicModeling />} />
              <Route path="/sentiment" element={<Sentiment />} />
              <Route path="/visualizations" element={<Visualizations />} />
              <Route path="/summarization" element={<Summarization />} />
              <Route path="/history" element={<History />} />
              <Route path="*" element={<NotFound />} />
            </Routes>
          </Layout>
        </BrowserRouter>
      </AnalysisProvider>
    </TooltipProvider>
  </QueryClientProvider>
);

export default App;
