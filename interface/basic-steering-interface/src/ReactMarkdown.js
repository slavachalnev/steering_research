import React, { useState } from "react";
import ReactMarkdown from "react-markdown";
import { unified } from "unified";
import remarkParse from "remark-parse";
import remarkStringify from "remark-stringify";
import "./App.css"; // Make sure to create a CSS file for styling

const MarkdownEditor = () => {
	const [markdown, setMarkdown] = useState(
		"**Hello world!** This is some sample text."
	);

	const tokenizeMarkdown = (text) => {
		const processor = unified().use(remarkParse).use(remarkStringify);
		const tree = processor.parse(text);
		return tree.children;
	};

	const tokens = tokenizeMarkdown(markdown);

	const renderToken = (token, index) => {
		const content =
			token.value ||
			(token.children && token.children.map((child) => child.value).join("")) ||
			"";
		const tokenClass = `token token-${index}`;

		return (
			<span
				key={index}
				className={tokenClass}
				onMouseOver={() => handleMouseOver(index)}
			>
				{content}
			</span>
		);
	};

	const handleMouseOver = (index) => {
		const tokenClass = `token-${index}`;
		document.querySelectorAll(`.${tokenClass}`).forEach((el) => {
			el.style.backgroundColor = "#ffff99"; // Highlight on hover
		});
	};

	return (
		<div>
			<textarea
				value={markdown}
				onChange={(e) => setMarkdown(e.target.value)}
				placeholder="Enter Markdown text"
				rows="10"
				cols="80"
			/>
			<div className="markdown-preview">
				{tokens.map((token, index) => renderToken(token, index))}
			</div>
		</div>
	);
};

export default MarkdownEditor;
