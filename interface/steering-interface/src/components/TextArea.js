import { useRef, useEffect } from "react";

export const AutoResizeTextArea = ({ value, onChange }) => {
	const textAreaRef = useRef(null);

	useEffect(() => {
		if (textAreaRef.current) {
			textAreaRef.current.style.height = "auto";
			textAreaRef.current.style.height = `${textAreaRef.current.scrollHeight}px`;
		}
	}, [value]);

	return (
		<textarea
			ref={textAreaRef}
			value={value}
			onChange={onChange}
			style={{ overflow: "hidden" }}
			rows="1"
			className="auto-resize-textarea"
		/>
	);
};
