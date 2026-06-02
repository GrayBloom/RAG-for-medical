"""
药物说明书 Markdown 解析器
将 MD 文件解析为结构化数据，支持字段级和段落级切分
"""
import re
import os
import json
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field, asdict


@dataclass
class DrugChunk:
    """文档切片——可以是字段级或段落级"""
    drug_name: str
    section: str           # 所属章节（药品名称/适应症/用法用量/...）
    content: str           # 文本内容
    chunk_id: str          # 唯一标识
    metadata: dict = field(default_factory=dict)


@dataclass
class DrugDocument:
    """完整的药物文档"""
    drug_name: str
    file_path: str
    sections: dict         # {section_name: content}
    chunks: list = field(default_factory=list)  # list of DrugChunk


class DrugMDParser:
    """解析药物说明书 Markdown 文件"""

    SECTION_NAMES = [
        "药品名称", "适应症", "用法用量", "规格", "禁忌",
        "注意事项", "不良反应", "药物相互作用", "药理作用",
        "药代动力学", "贮藏", "包装", "有效期", "生产企业",
        "成分", "性状", "孕妇及哺乳期妇女用药", "儿童用药",
        "老年用药", "药物过量", "执行标准", "批准文号",
    ]

    def __init__(self, md_dir: str):
        self.md_dir = Path(md_dir)
        self.documents: list[DrugDocument] = []

    def parse_all(self) -> list[DrugDocument]:
        """解析目录下所有 MD 文件"""
        for md_file in sorted(self.md_dir.glob("*.md")):
            doc = self._parse_file(md_file)
            if doc:
                self.documents.append(doc)
        return self.documents

    def _parse_file(self, file_path: Path) -> Optional[DrugDocument]:
        """解析单个 MD 文件"""
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        # 提取一级标题作为药品名
        title_match = re.match(r"^#\s+(.+?)\s*(?:说明书)?\s*$", text, re.MULTILINE)
        if not title_match:
            return None
        drug_name = title_match.group(1).strip().rstrip("说明书").strip()

        # 按二级标题切分各章节
        sections = {}
        pattern = r"^##\s+(.+?)\s*$"
        parts = re.split(pattern, text, flags=re.MULTILINE)

        # parts[0] 是标题行之前的内容（空或标题），之后是 章节名, 内容, 章节名, 内容...
        for i in range(1, len(parts), 2):
            section_name = parts[i].strip()
            section_content = parts[i + 1].strip() if i + 1 < len(parts) else ""
            sections[section_name] = section_content

        doc = DrugDocument(
            drug_name=drug_name,
            file_path=str(file_path),
            sections=sections,
        )

        # 生成切片
        doc.chunks = self._generate_chunks(doc, file_path.stem)
        return doc

    def _generate_chunks(self, doc: DrugDocument, file_stem: str) -> list[DrugChunk]:
        """为文档生成多种粒度的切片"""
        chunks = []

        # 1. 全文档摘要切片
        full_text = f"药品名称：{doc.drug_name}\n\n"
        for sec, content in doc.sections.items():
            full_text += f"{sec}：{content}\n\n"

        chunks.append(DrugChunk(
            drug_name=doc.drug_name,
            section="全文",
            content=full_text.strip(),
            chunk_id=f"{file_stem}__full",
            metadata={"type": "full_document"}
        ))

        # 2. 逐章节切片
        for sec, content in doc.sections.items():
            if not content.strip():
                continue
            chunks.append(DrugChunk(
                drug_name=doc.drug_name,
                section=sec,
                content=f"【{doc.drug_name}】{sec}：{content}",
                chunk_id=f"{file_stem}__{sec}",
                metadata={"type": "section", "section": sec}
            ))

        # 3. 适应症细粒度拆分（按序号）
        if "适应症" in doc.sections:
            indications = self._split_list(doc.sections["适应症"])
            for idx, item in enumerate(indications):
                if item.strip():
                    chunks.append(DrugChunk(
                        drug_name=doc.drug_name,
                        section="适应症",
                        content=f"【{doc.drug_name}】适应症：{item}",
                        chunk_id=f"{file_stem}__适应症__{idx}",
                        metadata={"type": "indication_item", "index": idx}
                    ))

        # 4. 用法用量细粒度拆分（按结构）
        if "用法用量" in doc.sections:
            dosages = self._split_dosage(doc.sections["用法用量"])
            for idx, item in enumerate(dosages):
                if item.strip():
                    chunks.append(DrugChunk(
                        drug_name=doc.drug_name,
                        section="用法用量",
                        content=f"【{doc.drug_name}】用法用量：{item}",
                        chunk_id=f"{file_stem}__用法用量__{idx}",
                        metadata={"type": "dosage_item", "index": idx}
                    ))

        return chunks

    def _split_list(self, text: str) -> list[str]:
        """拆分编号列表"""
        items = re.split(r"\d+[\.\、\s)]", text)
        items = [i.strip() for i in items if i.strip()]
        if not items:
            items = [text]
        return items

    def _split_dosage(self, text: str) -> list[str]:
        """拆分用法用量段落"""
        # 按空行或破折号拆分
        lines = text.split("\n")
        parts = []
        current = []
        for line in lines:
            line = line.strip()
            if not line:
                if current:
                    parts.append(" ".join(current))
                    current = []
            else:
                current.append(line.lstrip("- "))
        if current:
            parts.append(" ".join(current))
        return parts if parts else [text]

    def get_all_chunks(self) -> list[DrugChunk]:
        """获取所有文档的所有切片"""
        all_chunks = []
        for doc in self.documents:
            all_chunks.extend(doc.chunks)
        return all_chunks

    def export_json(self, output_path: str):
        """导出结构化数据"""
        data = []
        for doc in self.documents:
            data.append({
                "drug_name": doc.drug_name,
                "file_path": doc.file_path,
                "sections": doc.sections,
                "chunks": [asdict(c) for c in doc.chunks]
            })
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return output_path


if __name__ == "__main__":
    import sys
    md_dir = sys.argv[1] if len(sys.argv) > 1 else r"D:\Python_Program\RAG\cleaned_MD_optimized"
    parser = DrugMDParser(md_dir)
    docs = parser.parse_all()
    print(f"解析完成：{len(docs)} 份说明书")

    all_chunks = parser.get_all_chunks()
    print(f"生成切片：{len(all_chunks)} 个")

    for doc in docs:
        print(f"\n📄 {doc.drug_name}")
        for sec in doc.sections:
            preview = doc.sections[sec][:80].replace("\n", " ")
            print(f"  ├─ {sec}: {preview}...")
