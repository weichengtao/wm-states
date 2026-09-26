import { useRef, useState } from "react";
import * as Dialog from "@radix-ui/react-dialog";
import {
  ArrowRight,
  BookOpen,
  ChevronDown,
  CircleHelp,
  Search,
  X,
} from "lucide-react";
import { Button } from "./ui/button";
import GuideLink, { HelpLink } from "./GuideLink";
import {
  helpTopics,
  matchesHelp,
  pageHelp,
  stageHelp,
  stageMethodsPath,
  type WorkspacePage,
} from "@/lib/help";

export default function HelpPanel({ page }: { page: WorkspacePage }) {
  const [query, setQuery] = useState("");
  const searchInput = useRef<HTMLInputElement>(null);
  const context = pageHelp[page];
  const topics = helpTopics.filter((topic) =>
    matchesHelp(query, topic.title, topic.answer, topic.keywords),
  );
  const stages = stageHelp.filter((stage) =>
    matchesHelp(query, stage.id, stage.label, stage.summary),
  );
  const hasQuery = query.trim().length > 0;
  return (
    <Dialog.Root>
      <Dialog.Trigger asChild>
        <Button
          variant="ghost"
          size="sm"
          className="help-trigger"
          aria-label="Open pipeline help"
        >
          <CircleHelp size={17} />
          <span>Help</span>
        </Button>
      </Dialog.Trigger>
      <Dialog.Portal>
        <Dialog.Overlay className="help-overlay" />
        <Dialog.Content
          className="help-panel"
          onOpenAutoFocus={(event) => {
            event.preventDefault();
            searchInput.current?.focus();
          }}
        >
          <div className="help-panel-header">
            <span className="help-book-icon">
              <BookOpen size={21} />
            </span>
            <div>
              <div className="eyebrow">YOUR WORKSPACE GUIDE</div>
              <Dialog.Title>Pipeline help</Dialog.Title>
            </div>
            <Dialog.Close asChild>
              <Button
                variant="ghost"
                size="icon"
                aria-label="Close pipeline help"
              >
                <X size={19} />
              </Button>
            </Dialog.Close>
          </div>
          <Dialog.Description className="help-intro">
            Find a setting, understand a stage, or get back on track.
          </Dialog.Description>
          <div className="help-search-wrap">
            <div className="search-field help-search">
              <Search size={17} aria-hidden="true" />
              <input
                ref={searchInput}
                aria-label="Search pipeline help"
                placeholder="Search settings, methods, topics…"
                value={query}
                onChange={(event) => setQuery(event.target.value)}
              />
              {query && (
                <button
                  type="button"
                  aria-label="Clear help search"
                  onClick={() => {
                    setQuery("");
                    searchInput.current?.focus();
                  }}
                >
                  <X size={15} />
                </button>
              )}
            </div>
            {hasQuery && (
              <p className="help-search-count" role="status">
                {topics.length + stages.length} matching topics
              </p>
            )}
          </div>
          <div className="help-panel-scroll">
            {!hasQuery && (
              <section className="help-context">
                <span className="eyebrow">FOR THIS VIEW</span>
                <h3>{context.label}</h3>
                <p>{context.summary}</p>
                <GuideLink path={context.path}>Read the walkthrough</GuideLink>
              </section>
            )}
            {topics.length > 0 && (
              <section className="help-section">
                <h3>{hasQuery ? "Guidance" : "Common questions"}</h3>
                <div className="help-questions">
                  {topics.map((topic) => (
                    <details className="help-question" key={topic.title}>
                      <summary>
                        {topic.title}
                        <ChevronDown size={16} aria-hidden="true" />
                      </summary>
                      <div className="help-answer">
                        <p>{topic.answer}</p>
                        <GuideLink path={topic.path}>
                          Read more in the guide
                        </GuideLink>
                        {topic.reference && (
                          <div className="mt-2">
                            <HelpLink href={topic.reference.url}>
                              {topic.reference.label}
                            </HelpLink>
                          </div>
                        )}
                      </div>
                    </details>
                  ))}
                </div>
              </section>
            )}
            {stages.length > 0 && (
              <section className="help-section">
                <h3>
                  Methods by stage <span>{stages.length}</span>
                </h3>
                <div className="help-stage-list">
                  {stages.map((stage) => (
                    <GuideLink
                      key={stage.id}
                      path={stageMethodsPath(stage.id)}
                      className="help-stage-link"
                      label={`${stage.label} methods`}
                    >
                      <span>
                        <strong>{stage.label}</strong>
                        <small>{stage.summary}</small>
                      </span>
                    </GuideLink>
                  ))}
                </div>
              </section>
            )}
            {!topics.length && !stages.length && (
              <div className="help-no-results">
                <Search size={26} />
                <h3>No matching topics</h3>
                <p>
                  Try “null”, “figures”, or “cache”, or search the full guide.
                </p>
                <GuideLink path="">Open the full guide</GuideLink>
              </div>
            )}
          </div>
          <footer className="help-panel-footer">
            <GuideLink path="" className="help-full-guide">
              <BookOpen size={17} />
              Open the full pipeline guide
              <ArrowRight size={16} />
            </GuideLink>
            <p>Your settings and running jobs stay in place.</p>
          </footer>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
