"""
Report Generation Module
PDF and HTML report generation for analysis results

Generates:
- Polarimetric analysis reports
- Target detection reports
- Military intelligence reports
- Change detection reports
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import json
import base64
from io import BytesIO


@dataclass
class ReportSection:
    """A section in the report."""
    title: str
    content: str
    images: List[Dict[str, Any]] = None  # {'path': str, 'caption': str}
    tables: List[Dict[str, Any]] = None  # {'headers': [], 'rows': []}
    subsections: List['ReportSection'] = None


class ReportGenerator:
    """
    Generate PDF and HTML reports from analysis results.
    
    Uses reportlab for PDF generation.
    """
    
    def __init__(self, output_dir: Path):
        """
        Initialize report generator.
        
        Args:
            output_dir: Directory for output files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Report metadata
        self.title = "SAR Analysis Report"
        self.subtitle = ""
        self.author = "Multi-Sensor SAR Analysis Platform"
        self.date = datetime.utcnow()
        
        # Report sections
        self.sections: List[ReportSection] = []
    
    def set_title(self, title: str, subtitle: str = ""):
        """Set report title."""
        self.title = title
        self.subtitle = subtitle
    
    def add_section(self, section: ReportSection):
        """Add a section to the report."""
        self.sections.append(section)
    
    def add_overview_section(
        self,
        aoi_description: str,
        satellite: str,
        acquisition_date: datetime,
        analysis_modes: List[str],
        overview_image_path: Path = None
    ):
        """Add overview/introduction section."""
        content = f"""
This report presents the results of SAR analysis performed on {satellite} data
acquired on {acquisition_date.strftime('%Y-%m-%d %H:%M UTC')}.

**Area of Interest:** {aoi_description}

**Analysis Modes Applied:**
"""
        for mode in analysis_modes:
            content += f"• {mode}\n"
        
        images = []
        if overview_image_path and overview_image_path.exists():
            images.append({
                'path': str(overview_image_path),
                'caption': 'Overview of analysis area'
            })
        
        section = ReportSection(
            title="1. Overview",
            content=content,
            images=images if images else None
        )
        self.sections.append(section)
    
    def add_polarimetric_section(
        self,
        h_alpha_stats: Dict[str, float],
        freeman_durden_stats: Dict[str, float],
        pauli_image_path: Path = None,
        h_alpha_image_path: Path = None
    ):
        """Add polarimetric analysis section."""
        content = """
## Polarimetric Decomposition Results

The quad-pol data was processed using multiple decomposition techniques:

### H-α-A Decomposition (Cloude-Pottier)
Entropy (H), Alpha (α), and Anisotropy (A) parameters characterize the
scattering mechanisms present in the scene.

"""
        # Add statistics
        content += "**H-α Statistics:**\n"
        for key, value in h_alpha_stats.items():
            content += f"• {key}: {value:.3f}\n"
        
        content += """
### Freeman-Durden Decomposition
Three-component decomposition separating surface, double-bounce, and volume scattering.

"""
        content += "**Power Distribution:**\n"
        for key, value in freeman_durden_stats.items():
            content += f"• {key}: {value:.1f}%\n"
        
        images = []
        if pauli_image_path and pauli_image_path.exists():
            images.append({
                'path': str(pauli_image_path),
                'caption': 'Pauli RGB Composite (R: Double-bounce, G: Volume, B: Surface)'
            })
        if h_alpha_image_path and h_alpha_image_path.exists():
            images.append({
                'path': str(h_alpha_image_path),
                'caption': 'H-α Classification (9 zones)'
            })
        
        section = ReportSection(
            title="2. Polarimetric Analysis",
            content=content,
            images=images if images else None
        )
        self.sections.append(section)
    
    def add_detection_section(
        self,
        targets: List[Dict],
        detection_image_path: Path = None,
        include_chips: bool = True
    ):
        """Add target detection section."""
        # Count by type
        type_counts = {}
        for t in targets:
            ttype = t.get('target_type', 'unknown')
            type_counts[ttype] = type_counts.get(ttype, 0) + 1
        
        content = f"""
## Target Detection Results

A total of **{len(targets)}** targets were detected using CFAR and
polarimetric classification algorithms.

### Detection Summary:
"""
        for ttype, count in type_counts.items():
            content += f"• {ttype.capitalize()}: {count}\n"
        
        # Create targets table
        headers = ['ID', 'Type', 'Confidence', 'Size (m)', 'Location']
        rows = []
        for t in targets[:20]:  # Limit to 20 in main table
            rows.append([
                t.get('target_id', 'N/A'),
                t.get('target_type', 'unknown'),
                f"{t.get('confidence', 0):.2f}",
                f"{t.get('length_m', 0):.0f} x {t.get('width_m', 0):.0f}",
                f"{t.get('center_lat', 0):.4f}, {t.get('center_lon', 0):.4f}"
            ])
        
        images = []
        if detection_image_path and detection_image_path.exists():
            images.append({
                'path': str(detection_image_path),
                'caption': 'Target Detection Overlay'
            })
        
        section = ReportSection(
            title="3. Target Detection",
            content=content,
            images=images if images else None,
            tables=[{'headers': headers, 'rows': rows}] if rows else None
        )
        self.sections.append(section)
    
    def add_terrain_section(
        self,
        classification_stats: Dict[str, Dict],
        trafficability_stats: Dict[str, float] = None,
        classification_image_path: Path = None,
        trafficability_image_path: Path = None
    ):
        """Add terrain classification section."""
        content = """
## Terrain Classification Results

Land cover was classified using polarimetric signatures and H-α zone mapping.

### Land Cover Distribution:
"""
        for class_name, stats in classification_stats.items():
            pct = stats.get('percentage', 0)
            if pct > 0.1:  # Only show significant classes
                content += f"• {class_name}: {pct:.1f}%\n"
        
        if trafficability_stats:
            content += "\n### Trafficability Summary:\n"
            for key, value in trafficability_stats.items():
                content += f"• {key}: {value:.1f}%\n"
        
        images = []
        if classification_image_path and classification_image_path.exists():
            images.append({
                'path': str(classification_image_path),
                'caption': 'Terrain Classification Map'
            })
        if trafficability_image_path and trafficability_image_path.exists():
            images.append({
                'path': str(trafficability_image_path),
                'caption': 'Trafficability Map (Green=Easy, Red=Difficult)'
            })
        
        section = ReportSection(
            title="4. Terrain Classification",
            content=content,
            images=images if images else None
        )
        self.sections.append(section)
    
    def add_insar_section(
        self,
        coherence_stats: Dict[str, float],
        deformation_stats: Dict[str, float],
        change_stats: Dict[str, float],
        coherence_image_path: Path = None,
        deformation_image_path: Path = None,
        change_image_path: Path = None
    ):
        """Add InSAR/change detection section."""
        content = """
## InSAR Analysis Results

Interferometric analysis was performed to detect surface deformation and changes.

### Coherence Statistics:
"""
        for key, value in coherence_stats.items():
            content += f"• {key}: {value:.3f}\n"
        
        content += "\n### Deformation Statistics:\n"
        for key, value in deformation_stats.items():
            content += f"• {key}: {value:.3f}\n"
        
        content += "\n### Change Detection:\n"
        for key, value in change_stats.items():
            content += f"• {key}: {value:.1f}%\n"
        
        images = []
        if coherence_image_path and coherence_image_path.exists():
            images.append({
                'path': str(coherence_image_path),
                'caption': 'Coherence Map (1=Stable, 0=Changed)'
            })
        if deformation_image_path and deformation_image_path.exists():
            images.append({
                'path': str(deformation_image_path),
                'caption': 'Deformation Velocity (mm/year)'
            })
        if change_image_path and change_image_path.exists():
            images.append({
                'path': str(change_image_path),
                'caption': 'Change Detection Map'
            })
        
        section = ReportSection(
            title="5. InSAR & Change Detection",
            content=content,
            images=images if images else None
        )
        self.sections.append(section)
    
    def add_military_summary(
        self,
        threat_assessment: Dict[str, Any],
        key_findings: List[str],
        recommendations: List[str]
    ):
        """Add military intelligence summary section."""
        content = """
## Military Intelligence Summary

### Threat Assessment:
"""
        for key, value in threat_assessment.items():
            content += f"• **{key}:** {value}\n"
        
        content += "\n### Key Findings:\n"
        for i, finding in enumerate(key_findings, 1):
            content += f"{i}. {finding}\n"
        
        content += "\n### Recommendations:\n"
        for i, rec in enumerate(recommendations, 1):
            content += f"{i}. {rec}\n"
        
        section = ReportSection(
            title="6. Intelligence Summary",
            content=content
        )
        self.sections.append(section)
    
    def generate_pdf(self, filename: str = "analysis_report.pdf") -> Path:
        """
        Generate PDF report.
        
        Args:
            filename: Output filename
            
        Returns:
            Path to generated PDF
        """
        try:
            from reportlab.lib import colors
            from reportlab.lib.pagesizes import letter, A4
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            from reportlab.platypus import (
                SimpleDocTemplate, Paragraph, Spacer, Image, Table, 
                TableStyle, PageBreak
            )
            from reportlab.lib.enums import TA_CENTER, TA_LEFT
        except ImportError:
            # Fall back to HTML if reportlab not available
            return self.generate_html(filename.replace('.pdf', '.html'))
        
        output_path = self.output_dir / filename
        
        doc = SimpleDocTemplate(
            str(output_path),
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )
        
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            alignment=TA_CENTER,
            spaceAfter=30
        )
        
        subtitle_style = ParagraphStyle(
            'CustomSubtitle',
            parent=styles['Normal'],
            fontSize=14,
            alignment=TA_CENTER,
            textColor=colors.grey,
            spaceAfter=20
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            spaceBefore=20,
            spaceAfter=10
        )
        
        body_style = ParagraphStyle(
            'CustomBody',
            parent=styles['Normal'],
            fontSize=11,
            spaceBefore=6,
            spaceAfter=6
        )
        
        # Build document content
        story = []
        
        # Title page
        story.append(Spacer(1, 2*inch))
        story.append(Paragraph(self.title, title_style))
        if self.subtitle:
            story.append(Paragraph(self.subtitle, subtitle_style))
        story.append(Spacer(1, 0.5*inch))
        story.append(Paragraph(f"Generated: {self.date.strftime('%Y-%m-%d %H:%M UTC')}", subtitle_style))
        story.append(Paragraph(f"By: {self.author}", subtitle_style))
        story.append(PageBreak())
        
        # Table of contents placeholder
        story.append(Paragraph("Table of Contents", heading_style))
        for section in self.sections:
            story.append(Paragraph(f"• {section.title}", body_style))
        story.append(PageBreak())
        
        # Sections
        for section in self.sections:
            # Section title
            story.append(Paragraph(section.title, heading_style))
            
            # Content (convert markdown-like to paragraphs)
            for para in section.content.strip().split('\n\n'):
                para = para.strip()
                if para:
                    # Handle bold
                    para = para.replace('**', '<b>').replace('<b>', '</b>', 1)
                    story.append(Paragraph(para.replace('\n', '<br/>'), body_style))
            
            # Images
            if section.images:
                for img_info in section.images:
                    try:
                        img_path = img_info['path']
                        if Path(img_path).exists():
                            img = Image(img_path, width=5*inch, height=3.5*inch)
                            story.append(Spacer(1, 0.2*inch))
                            story.append(img)
                            if img_info.get('caption'):
                                story.append(Paragraph(
                                    f"<i>{img_info['caption']}</i>",
                                    ParagraphStyle('Caption', parent=body_style, 
                                                 alignment=TA_CENTER, fontSize=10)
                                ))
                    except Exception as e:
                        print(f"Could not add image: {e}")
            
            # Tables
            if section.tables:
                for tbl_info in section.tables:
                    headers = tbl_info.get('headers', [])
                    rows = tbl_info.get('rows', [])
                    
                    if headers and rows:
                        data = [headers] + rows
                        table = Table(data)
                        table.setStyle(TableStyle([
                            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                            ('FONTSIZE', (0, 0), (-1, -1), 9),
                            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                            ('GRID', (0, 0), (-1, -1), 1, colors.black)
                        ]))
                        story.append(Spacer(1, 0.2*inch))
                        story.append(table)
            
            story.append(Spacer(1, 0.3*inch))
        
        # Build PDF
        doc.build(story)
        
        return output_path
    
    def generate_html(self, filename: str = "analysis_report.html") -> Path:
        """
        Generate HTML report.
        
        Args:
            filename: Output filename
            
        Returns:
            Path to generated HTML
        """
        output_path = self.output_dir / filename
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.title}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 900px;
            margin: 0 auto;
            padding: 40px 20px;
            line-height: 1.6;
            color: #333;
        }}
        h1 {{ color: #1a365d; border-bottom: 3px solid #3182ce; padding-bottom: 10px; }}
        h2 {{ color: #2c5282; margin-top: 40px; }}
        h3 {{ color: #4a5568; }}
        .subtitle {{ color: #718096; font-size: 1.1em; }}
        .meta {{ color: #a0aec0; font-size: 0.9em; margin-bottom: 30px; }}
        img {{ max-width: 100%; height: auto; border-radius: 8px; margin: 20px 0; }}
        .caption {{ text-align: center; color: #718096; font-style: italic; margin-top: -10px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #e2e8f0; padding: 12px; text-align: left; }}
        th {{ background: #edf2f7; font-weight: 600; }}
        tr:nth-child(even) {{ background: #f7fafc; }}
        .toc {{ background: #f7fafc; padding: 20px; border-radius: 8px; margin: 30px 0; }}
        .toc ul {{ margin: 0; padding-left: 20px; }}
        .section {{ margin-bottom: 40px; }}
        code {{ background: #edf2f7; padding: 2px 6px; border-radius: 4px; }}
    </style>
</head>
<body>
    <h1>{self.title}</h1>
"""
        
        if self.subtitle:
            html += f'    <p class="subtitle">{self.subtitle}</p>\n'
        
        html += f"""    <p class="meta">Generated: {self.date.strftime('%Y-%m-%d %H:%M UTC')} | By: {self.author}</p>
    
    <div class="toc">
        <h3>Table of Contents</h3>
        <ul>
"""
        
        for section in self.sections:
            anchor = section.title.lower().replace(' ', '-').replace('.', '')
            html += f'            <li><a href="#{anchor}">{section.title}</a></li>\n'
        
        html += """        </ul>
    </div>
"""
        
        for section in self.sections:
            anchor = section.title.lower().replace(' ', '-').replace('.', '')
            html += f'\n    <div class="section" id="{anchor}">\n'
            html += f'        <h2>{section.title}</h2>\n'
            
            # Process content
            content = section.content.strip()
            content = content.replace('**', '<strong>').replace('<strong>', '</strong>', 1)
            
            for para in content.split('\n\n'):
                para = para.strip()
                if para.startswith('###'):
                    html += f'        <h3>{para.replace("### ", "")}</h3>\n'
                elif para.startswith('##'):
                    html += f'        <h3>{para.replace("## ", "")}</h3>\n'
                elif para:
                    html += f'        <p>{para.replace(chr(10), "<br>")}</p>\n'
            
            # Images
            if section.images:
                for img_info in section.images:
                    img_path = img_info['path']
                    caption = img_info.get('caption', '')
                    
                    # Try to embed as base64
                    try:
                        if Path(img_path).exists():
                            with open(img_path, 'rb') as f:
                                img_data = base64.b64encode(f.read()).decode()
                            suffix = Path(img_path).suffix.lower()
                            mime = {'png': 'image/png', 'jpg': 'image/jpeg', 
                                   'jpeg': 'image/jpeg', 'gif': 'image/gif'}.get(suffix[1:], 'image/png')
                            html += f'        <img src="data:{mime};base64,{img_data}" alt="{caption}">\n'
                            if caption:
                                html += f'        <p class="caption">{caption}</p>\n'
                    except:
                        html += f'        <p><em>Image: {img_path}</em></p>\n'
            
            # Tables
            if section.tables:
                for tbl_info in section.tables:
                    headers = tbl_info.get('headers', [])
                    rows = tbl_info.get('rows', [])
                    
                    if headers:
                        html += '        <table>\n'
                        html += '            <tr>' + ''.join(f'<th>{h}</th>' for h in headers) + '</tr>\n'
                        for row in rows:
                            html += '            <tr>' + ''.join(f'<td>{c}</td>' for c in row) + '</tr>\n'
                        html += '        </table>\n'
            
            html += '    </div>\n'
        
        html += """
</body>
</html>"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        return output_path
    
    def generate_markdown(self, filename: str = "analysis_report.md") -> Path:
        """Generate Markdown report."""
        output_path = self.output_dir / filename
        
        md = f"# {self.title}\n\n"
        
        if self.subtitle:
            md += f"*{self.subtitle}*\n\n"
        
        md += f"**Generated:** {self.date.strftime('%Y-%m-%d %H:%M UTC')}  \n"
        md += f"**By:** {self.author}\n\n"
        md += "---\n\n"
        
        # TOC
        md += "## Table of Contents\n\n"
        for section in self.sections:
            anchor = section.title.lower().replace(' ', '-').replace('.', '')
            md += f"- [{section.title}](#{anchor})\n"
        md += "\n---\n\n"
        
        # Sections
        for section in self.sections:
            md += f"## {section.title}\n\n"
            md += section.content.strip() + "\n\n"
            
            if section.images:
                for img_info in section.images:
                    md += f"![{img_info.get('caption', '')}]({img_info['path']})\n"
                    if img_info.get('caption'):
                        md += f"*{img_info['caption']}*\n\n"
            
            if section.tables:
                for tbl_info in section.tables:
                    headers = tbl_info.get('headers', [])
                    rows = tbl_info.get('rows', [])
                    
                    if headers:
                        md += "| " + " | ".join(headers) + " |\n"
                        md += "| " + " | ".join(['---'] * len(headers)) + " |\n"
                        for row in rows:
                            md += "| " + " | ".join(str(c) for c in row) + " |\n"
                        md += "\n"
            
            md += "\n"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(md)
        
        return output_path
