export const fetchData = async (index) => {
	console.log("fetching: " + index);
	try {
		const response = await fetch(
			`http://localhost:5000/get_data?index=${index}&type=cosine`
		);
		if (!response.ok) {
			throw new Error("Network response was not ok");
		}
		const rawData = await response.json();
		return rawData;
	} catch (error) {
		console.error("There was a problem with the fetch operation:", error);
	}
};
