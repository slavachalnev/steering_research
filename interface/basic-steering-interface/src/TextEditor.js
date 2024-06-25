import React, { useState } from "react";
import CodeMirror from "@uiw/react-codemirror";
import { markdown, markdownLanguage } from "@codemirror/lang-markdown";
// import "codemirror/lib/codemirror.css";
// import "codemirror/mode/javascript/javascript"; // or any other mode you prefer

const TextEditor = () => {
	const [text, setText] = useState("");
	const [tokens, setTokens] = useState([]);

	const handleTextChange = (editor, data, value) => {
		setText(value);
		// Send the text to the backend to get tokenized
		fetch("/tokenize", {
			method: "POST",
			headers: {
				"Content-Type": "application/json",
			},
			body: JSON.stringify({ text: value }),
		})
			.then((response) => response.json())
			.then((data) => {
				setTokens(data.tokens);
			});
	};

	const highlightToken = (tokenIndex) => {
		// Logic to highlight the token in the editor
	};

	return (
		<div>
			<CodeMirror
				value={text}
				extensions={[markdown({ base: markdownLanguage })]}
				onBeforeChange={(editor, data, value) => {
					setText(value);
				}}
				onChange={handleTextChange}
			/>
			<div>
				{tokens.map((token, index) => (
					<span
						key={index}
						onClick={() => highlightToken(index)}
						style={{ cursor: "pointer" }}
					>
						{token}
					</span>
				))}
			</div>
		</div>
	);
};

export default TextEditor;
