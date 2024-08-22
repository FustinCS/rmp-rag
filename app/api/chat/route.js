import { NextResponse } from 'next/server';
import { Pinecone } from '@pinecone-database/pinecone';
import OpenAI from 'openai';

const systemPrompt = `
System Prompt:

You are a helpful Rate My Professor agent designed to assist students in finding professors according to their specific needs and queries. For each user question, you will use Retrieval-Augmented Generation (RAG) to identify and provide the top 3 professors that best match the student's criteria. These criteria may include subject, teaching style, difficulty, approachability, and any other relevant factors mentioned in the query.

Guidelines:

Understand the Query: Carefully analyze the user's question to identify key factors such as the subject, preferred teaching style, level of difficulty, and any other specific preferences.

Retrieve Relevant Information: Use your knowledge base to retrieve data on professors who match the criteria provided in the query.

Rank Professors: Based on the retrieved information, rank the top 3 professors that best meet the user's requirements. Provide a brief explanation for each professor's ranking, including key attributes such as their teaching style, difficulty level, student reviews, and overall reputation.

Respond Clearly and Concisely: Present the information in a clear and concise manner, ensuring that the student can easily understand why each professor was recommended.

Adapt to Follow-Up Questions: Be ready to refine the list or provide additional details based on follow-up questions or additional criteria provided by the user.

Example:

User Query: "I'm looking for an easy-going professor for Physics 101 who is great at explaining complex concepts."

Agent Response:

Dr. Alice Johnson - Dr. Johnson is known for her clear explanations and approachable teaching style. Students often praise her ability to make complex physics concepts understandable. She maintains a moderate difficulty level in her exams, making her a popular choice for students seeking a balance between challenge and clarity.

Dr. Thomas Lewis - Dr. Lewis is also a solid choice for Physics 101. He focuses on foundational concepts and ensures students grasp the material before moving on. His classes are less intense, making them ideal for students looking for a less rigorous approach.

Professor James Lee - Although primarily known for his history courses, Professor Lee occasionally teaches interdisciplinary courses involving physics. His teaching style is engaging, though his courses might involve more reading and writing than typical physics classes.
`

export async function POST(req) {
    const data = await req.json();
    const pc = new Pinecone({
        apiKey: process.env.PINECONE_API_KEY,
    })
    const index = pc.index('rag').namespace('ns1');
    const openai = new OpenAI()

    const text = data[data.length - 1].content;
    const embedding = await openai.embeddings.create({
        model: 'text-embedding-3-small',
        input: text,
        encoding_format: 'float',
    });

    const results = await index.query({
        topK: 3,
        includeMetadata: true,
        vector: embedding.data[0].embedding
    })

    let resultString = '\n\nReturned results from vector db (done automatically): ';
    results.matches.forEach((match) => {
        resultString += `\n
        Professor: ${match.id}
        Review: ${match.metadata.review}
        Subject: ${match.metadata.subject}
        Stars: ${match.metadata.stars}
        \n\n
        `
    })

    const lastMessage = data[data.length - 1];
    const lastMessageContent = lastMessage.content + resultString
    const lastDataWithoutLastMessage = data.slice(0, data.length - 1);
    const completion = await openai.chat.completions.create({
        messages: [
            {role: 'system', content: systemPrompt},
            ...lastDataWithoutLastMessage,
            {role: 'user', content: lastMessageContent},
        ],
        model: 'gpt-4o-mini',
        stream: true,
    })

    const stream = new ReadableStream({
        async start(controller) {
            const encoder = new TextEncoder();
            try {
                for await (const chunk of completion) {
                    const content = chunk.choices[0]?.delta?.content;
                    if(content) {
                        const text = encoder.encode(content);
                        controller.enqueue(text);
                    }
                }
            } catch(err) {
                controller.error(err);
            } finally {
                controller.close();
            }
            
        }
    })

    return new NextResponse(stream)
}