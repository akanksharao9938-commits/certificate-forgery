import { useEffect, useMemo, useState } from "react";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

function Icon({ name, size = 20 }) {
  const common = { width:size, height:size, viewBox:"0 0 24 24", fill:"none",
    stroke:"currentColor", strokeWidth:1.8, strokeLinecap:"round", strokeLinejoin:"round" };
  const p = {
    shield:<><path d="M12 3l7 3v5c0 4.8-3 8.4-7 10-4-1.6-7-5.2-7-10V6l7-3z"/><path d="m9 12 2 2 4-4"/></>,
    upload:<><path d="M12 16V4"/><path d="m7 9 5-5 5 5"/><path d="M5 20h14"/></>,
    scan:<><path d="M4 7V5a1 1 0 0 1 1-1h2"/><path d="M17 4h2a1 1 0 0 1 1 1v2"/><path d="M20 17v2a1 1 0 0 1-1 1h-2"/><path d="M7 20H5a1 1 0 0 1-1-1v-2"/><circle cx="12" cy="12" r="3.5"/></>,
    file:<><path d="M6 3h8l4 4v14H6z"/><path d="M14 3v5h5"/><path d="M9 13h6"/><path d="M9 17h4"/></>,
    check:<path d="m5 12 4 4L19 6"/>,
    alert:<><path d="M12 3 2.8 20h18.4L12 3z"/><path d="M12 9v4"/><path d="M12 17h.01"/></>,
    cpu:<><rect x="7" y="7" width="10" height="10" rx="1"/><path d="M9 1v3M15 1v3M9 20v3M15 20v3M20 9h3M20 14h3M1 9h3M1 14h3"/></>,
    text:<><path d="M4 6h16M8 6v13M16 6v13M6 19h12"/></>,
    activity:<path d="M3 12h4l2-7 4 14 2-7h6"/>,
    refresh:<><path d="M20 11a8 8 0 0 0-14-5L4 8"/><path d="M4 4v4h4"/><path d="M4 13a8 8 0 0 0 14 5l2-2"/><path d="M20 20v-4h-4"/></>,
    external:<><path d="M14 4h6v6"/><path d="M20 4 10 14"/><path d="M18 13v6a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V7a1 1 0 0 1 1-1h6"/></>
  };
  return <svg {...common}>{p[name]}</svg>;
}

function formatPercent(v) {
  if (v === undefined || v === null || v === "") return "—";
  const n = Number(v);
  return Number.isFinite(n) ? `${n.toFixed(2)}%` : String(v);
}
function formatProbability(v) {
  if (v === undefined || v === null || v === "") return "—";
  const n = Number(v);
  return Number.isFinite(n) ? `${(n * 100).toFixed(2)}%` : String(v);
}

function MetricCard({ icon, title, value, rows }) {
  return <div className="card metric-card">
    <div className="metric-top"><div className="metric-icon"><Icon name={icon} size={19}/></div><span>{title}</span></div>
    <h3>{value}</h3>
    <div className="metric-rows">{rows.map(([label,val],i)=><div key={i}><span>{label}</span><strong>{val ?? "—"}</strong></div>)}</div>
  </div>;
}

export default function App() {
  const api = API_BASE_URL.replace(/\/$/, "");
  const [file,setFile]=useState(null), [preview,setPreview]=useState("");
  const [result,setResult]=useState(null), [loading,setLoading]=useState(false);
  const [health,setHealth]=useState("checking"), [error,setError]=useState("");
  const [dragging,setDragging]=useState(false);

  const resultImage = useMemo(() => {
    if (!result?.result_image) return "";
    return result.result_image.startsWith("http") ? result.result_image : `${api}${result.result_image}`;
  }, [result,api]);

  const checkHealth = async () => {
    try {
      setHealth("checking");
      const r=await fetch(`${api}/health`);
      if(!r.ok) throw new Error();
      const d=await r.json();
      setHealth(d.status==="healthy" ? "online":"offline");
    } catch { setHealth("offline"); }
  };
  useEffect(()=>{checkHealth(); const t=setInterval(checkHealth,30000); return()=>clearInterval(t)},[]);

  const chooseFile = f => {
    if(!f) return;
    if(!f.type.startsWith("image/")) { setError("Please upload a JPG, JPEG, or PNG image."); return; }
    if(preview) URL.revokeObjectURL(preview);
    setFile(f); setPreview(URL.createObjectURL(f)); setResult(null); setError("");
  };
  const analyze = async () => {
    if(!file){setError("Choose a certificate image before starting the analysis.");return;}
    setLoading(true);setError("");setResult(null);
    try{
      const fd=new FormData(); fd.append("file",file);
      const r=await fetch(`${api}/api/predict`,{method:"POST",body:fd});
      const d=await r.json();
      if(!r.ok) throw new Error(typeof d.detail==="string"?d.detail:"The backend could not analyze this certificate.");
      setResult(d);
    }catch(e){setError(e.message||"Could not connect to the certificate analysis service.");}
    finally{setLoading(false);}
  };
  const reset=()=>{if(preview)URL.revokeObjectURL(preview);setFile(null);setPreview("");setResult(null);setError("")};
  const authentic=result?.prediction==="Authentic";
  const confidence=typeof result?.confidence==="number"?result.confidence:0;

  return <div className="app-shell">
    <header className="topbar">
      <div className="brand"><div className="brand-mark"><Icon name="shield" size={24}/></div><div><div className="brand-name">CertificateGuard AI</div><div className="brand-subtitle">Academic certificate analysis</div></div></div>
      <div className={`backend-pill ${health}`}><span className="status-dot"/>{health==="online"?"AI service online":health==="checking"?"Checking service...":"AI service offline"}</div>
    </header>

    <main>
      <section className="hero-section">
        <div className="eyebrow"><Icon name="scan" size={16}/> MULTI-MODEL DOCUMENT ANALYSIS</div>
        <h1>Verify an academic certificate with AI.</h1>
        <p>Upload a certificate image and get a combined analysis from ResNet50 classification, YOLO11 region detection, OCR + LOF anomaly analysis, and pixel-level image statistics.</p>
        <div className="model-row">
          <span><Icon name="cpu" size={15}/> ResNet50</span><span><Icon name="scan" size={15}/> YOLO11</span><span><Icon name="text" size={15}/> OCR + LOF</span><span><Icon name="activity" size={15}/> Pixel analysis</span>
        </div>
      </section>

      <section className="workspace">
        <div className="upload-panel card">
          <div className="section-heading"><div><span className="step">STEP 01</span><h2>Upload certificate</h2></div>{file&&<button className="text-button" onClick={reset}>Remove</button>}</div>
          <label className={`drop-zone ${dragging?"dragging":""} ${file?"has-file":""}`}
            onDragOver={e=>{e.preventDefault();setDragging(true)}} onDragLeave={()=>setDragging(false)}
            onDrop={e=>{e.preventDefault();setDragging(false);chooseFile(e.dataTransfer.files?.[0])}}>
            <input type="file" accept="image/png,image/jpeg,image/jpg" onChange={e=>chooseFile(e.target.files?.[0])}/>
            {preview?<img src={preview} alt="Selected certificate preview" className="preview"/>:<><div className="upload-icon"><Icon name="upload" size={30}/></div><strong>Drop your certificate here</strong><span>or click to browse from your computer</span><small>JPG, JPEG or PNG</small></>}
          </label>
          {file&&<div className="file-row"><div className="file-icon"><Icon name="file" size={19}/></div><div className="file-info"><strong>{file.name}</strong><span>{(file.size/1024/1024).toFixed(2)} MB</span></div><Icon name="check" size={19}/></div>}
          <button className="primary-button" onClick={analyze} disabled={!file||loading}>{loading?<><span className="spinner"/>Analyzing certificate...</>:<><Icon name="scan" size={19}/>Analyze certificate</>}</button>
          {error&&<div className="error-message"><Icon name="alert" size={19}/><span>{error}</span></div>}
        </div>

        <div className="workflow-panel"><div className="workflow-card card">
          <span className="step">WHAT HAPPENS NEXT</span><h3>Four analysis layers</h3>
          {[["01","Whole-image classification","ResNet50 estimates Authentic vs Forged."],["02","Suspicious regions","YOLO11 identifies detected fake/true regions."],["03","Text & spatial anomalies","OCR extracts words and LOF checks local patterns."],["04","Image statistics","Edge density and pixel noise are reported."]].map(x=><div className="workflow-item" key={x[0]}><div className="workflow-number">{x[0]}</div><div><strong>{x[1]}</strong><p>{x[2]}</p></div></div>)}
        </div></div>
      </section>

      {result&&<section className="results-section">
        <div className="results-header"><div><span className="step">STEP 02</span><h2>Analysis result</h2></div><button className="secondary-button" onClick={reset}><Icon name="refresh" size={17}/>Analyze another</button></div>

        <div className={`verdict-card ${authentic?"authentic":"forged"}`}>
          <div className="verdict-icon"><Icon name={authentic?"check":"alert"} size={34}/></div>
          <div className="verdict-copy"><span>OVERALL CLASSIFICATION</span><h3>{result.prediction||"Unknown"}</h3><p>{result.message||result.interpretation}</p></div>
          <div className="confidence-ring" style={{background:`conic-gradient(#2563eb ${confidence*3.6}deg,#e2e8f0 0deg)`}}><div><strong>{confidence.toFixed(1)}%</strong><span>confidence</span></div></div>
        </div>

        <div className="metrics-grid">
          <MetricCard icon="cpu" title="ResNet50" value={result.classification?.prediction||"—"} rows={[["Confidence",formatPercent(result.classification?.confidence)],["Forged probability",formatProbability(result.classification?.forged_probability)],["Authentic probability",formatProbability(result.classification?.authentic_probability)]]}/>
          <MetricCard icon="scan" title="YOLO11 regions" value={`${result.region_detection?.fake_region_count??0} suspicious`} rows={[["True regions",result.region_detection?.true_region_count??0],["Fake confidence",formatPercent(result.region_detection?.fake_region_confidence)],["True confidence",formatPercent(result.region_detection?.true_region_confidence)]]}/>
          <MetricCard icon="text" title="OCR + LOF" value={`${result.text_analysis?.anomaly_count??0} anomalies`} rows={[["OCR words",result.text_analysis?.ocr_word_count??0],["Anomaly rate",formatPercent(result.text_analysis?.anomaly_rate_percent)],["Features",result.text_analysis?.spatial_density_features?.feature_count??0]]}/>
          <MetricCard icon="activity" title="Pixel analysis" value={formatPercent(result.pixel_analysis?.edge_density_percent)} rows={[["Edge density",formatPercent(result.pixel_analysis?.edge_density_percent)],["Pixel noise score",result.pixel_analysis?.pixel_noise_score??"—"]]}/>
        </div>

        <div className="two-column">
          <div className="card result-card"><div className="card-title"><div><span className="step">MODEL INTERPRETATION</span><h3>What the system found</h3></div></div>
            <p className="interpretation">{result.interpretation||"No interpretation returned."}</p>
            {result.region_detection?.summary&&<div className="info-line"><span>YOLO11 summary</span><strong>{result.region_detection.summary}</strong></div>}
            <div className="notice"><Icon name="alert" size={18}/><span>A detected anomaly or suspicious region is an indicator for review, not by itself proof that a certificate is forged.</span></div>
          </div>

          <div className="card result-card"><div className="card-title"><div><span className="step">TEXT ANALYSIS</span><h3>OCR anomalies</h3></div></div>
            {result.text_analysis?.anomalies?.length?<div className="anomaly-list">{result.text_analysis.anomalies.map((a,i)=><div className="anomaly-item" key={i}><div><span>TEXT</span><strong>{a.text||"Unreadable"}</strong></div><div><span>OCR confidence</span><strong>{a.ocr_confidence??"—"}</strong></div><div><span>LOF score</span><strong>{a.lof_score??"—"}</strong></div></div>)}</div>:<div className="empty-state"><Icon name="check" size={20}/><span>{result.text_analysis?.message||"No significant local typographical or spatial anomalies detected."}</span></div>}
          </div>
        </div>

        <div className="card annotated-card"><div className="card-title"><div><span className="step">VISUAL EVIDENCE</span><h3>Annotated detection result</h3></div>{resultImage&&<a className="secondary-button link-button" href={resultImage} target="_blank" rel="noreferrer"><Icon name="external" size={16}/>Open image</a>}</div>
          {resultImage?<div className="annotated-image-wrap"><img src={resultImage} alt="Annotated certificate result" className="annotated-image"/></div>:<div className="empty-state">No result image returned.</div>}
        </div>

        {result.region_detection?.detections?.length>0&&<div className="card table-card"><div className="card-title"><div><span className="step">YOLO11 DETAILS</span><h3>Detected regions</h3></div></div>
          <div className="table-wrap"><table><thead><tr><th>Class</th><th>Confidence</th><th>Bounding box</th></tr></thead><tbody>{result.region_detection.detections.map((d,i)=><tr key={i}><td><span className={`class-badge ${d.class==="fake"?"fake":"true"}`}>{d.class}</span></td><td>{formatPercent(d.confidence)}</td><td>{d.bounding_box?`${d.bounding_box.x1}, ${d.bounding_box.y1} → ${d.bounding_box.x2}, ${d.bounding_box.y2}`:"—"}</td></tr>)}</tbody></table></div>
        </div>}
      </section>}
    </main>

    <footer><div><Icon name="shield" size={19}/><strong>CertificateGuard AI</strong></div><span>AI-assisted academic certificate authenticity analysis</span></footer>
  </div>;
}