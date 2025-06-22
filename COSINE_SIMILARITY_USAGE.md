# Cosine Similarity Usage in Multimodal RAG

## Overview

This document explains how cosine similarity is used in our multimodal RAG system and addresses the question about the `calculate_cosine_similarity` function.

## Why the Original Function Wasn't Used

### The Issue
The original `calculate_cosine_similarity` function was defined but **never actually called** in the codebase. This happened because:

1. **Databricks Vector Search handles similarity internally** - When you call `index.similarity_search()`, Databricks calculates cosine similarity automatically
2. **No manual similarity calculation needed** - The vector search index returns pre-calculated similarity scores
3. **Function was redundant** - It was implementing something that Databricks already does

### What Databricks Vector Search Does
```python
# This automatically calculates cosine similarity internally
results = index.similarity_search(
    query_vector=query_embedding,
    num_results=5
)

# Results come back with similarity scores already calculated
for result in results:
    score = result[4]  # Similarity score from Databricks
    print(f"Similarity: {score}")
```

## Current Implementation

### Where Cosine Similarity IS Used

We now use cosine similarity in **validation and debugging**:

```python
def validate_similarity_scores(query_embedding, results):
    """
    Validate that Databricks Vector Search scores are correct
    by calculating cosine similarity manually
    """
    for result in results:
        # Get embeddings
        query_array = np.array(query_embedding)
        stored_array = np.array(result['embeddings'])
        
        # Normalize vectors
        query_normalized = query_array / np.linalg.norm(query_array)
        stored_normalized = stored_array / np.linalg.norm(stored_array)
        
        # Calculate cosine similarity manually
        calculated_similarity = np.dot(query_normalized, stored_normalized)
        
        # Compare with Databricks score
        vector_search_score = result['similarity_score']
        difference = abs(calculated_similarity - vector_search_score)
        
        print(f"Databricks: {vector_search_score:.4f}")
        print(f"Calculated: {calculated_similarity:.4f}")
        print(f"Difference: {difference:.4f}")
```

### How to Use Validation

```python
# Enable validation for debugging
similar_images = perform_optimized_similarity_search(
    query="¿Que es un CDV?",
    num_results=3,
    validate_scores=True  # This will validate similarity scores
)
```

## Similarity Calculation Methods

### 1. Databricks Vector Search (Primary)
```python
# This is what we use for actual similarity search
results = index.similarity_search(
    query_vector=query_embedding,
    num_results=5
)
```

### 2. Manual Validation (Debugging)
```python
# This is what we use to validate Databricks scores
def validate_similarity_scores(query_embedding, results):
    # Calculate cosine similarity manually
    # Compare with Databricks scores
    # Report any discrepancies
```

### 3. Direct Comparison (Advanced)
```python
# For comparing two embeddings directly
def compare_embeddings(embedding1, embedding2):
    # Normalize vectors
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    
    # Calculate cosine similarity
    similarity = np.dot(embedding1/norm1, embedding2/norm2)
    return similarity
```

## Why This Approach is Better

### 1. **Performance**
- Databricks Vector Search is optimized for large-scale similarity search
- Manual calculation would be much slower for large datasets

### 2. **Accuracy**
- Databricks uses the same cosine similarity formula
- Our validation ensures scores are correct

### 3. **Scalability**
- Vector search index can handle millions of embeddings
- Manual calculation would not scale

### 4. **Debugging**
- Validation helps identify issues with embeddings or scores
- Useful for troubleshooting similarity problems

## Usage Examples

### Normal Search (No Validation)
```python
# Standard similarity search
results = perform_optimized_similarity_search(
    query="¿Que es un CDV?",
    num_results=5
)
```

### Search with Validation
```python
# Search with similarity score validation
results = perform_optimized_similarity_search(
    query="¿Que es un CDV?",
    num_results=5,
    validate_scores=True  # Will validate each score
)
```

### Manual Validation Only
```python
# Validate existing results
from vector_search_setup import validate_similarity_scores

validated_results = validate_similarity_scores(
    query_embedding=my_embedding,
    results=my_results
)
```

## Validation Output Example

When validation is enabled, you'll see output like:

```
Validating Similarity Scores
==========================================================
PDF: /path/to/doc.pdf, Page: 1
  Vector Search Score: 0.8234
  Calculated Similarity: 0.8234
  Difference: 0.0000
  Valid: True

PDF: /path/to/doc2.pdf, Page: 3
  Vector Search Score: 0.7567
  Calculated Similarity: 0.7567
  Difference: 0.0000
  Valid: True

Validation Summary:
Valid scores: 2/2
✓ All similarity scores are valid!
```

## Best Practices

### 1. **Use Databricks Vector Search for Production**
- Always use `index.similarity_search()` for actual queries
- It's optimized and handles large datasets efficiently

### 2. **Use Validation for Debugging**
- Enable `validate_scores=True` when troubleshooting
- Helps identify issues with embeddings or scores

### 3. **Monitor Score Differences**
- Large differences (>0.01) may indicate issues
- Check embedding normalization if problems occur

### 4. **Don't Calculate Manually in Production**
- Manual calculation is slow and unnecessary
- Use validation only for debugging

## Conclusion

The original `calculate_cosine_similarity` function wasn't used because:

1. **Databricks Vector Search already calculates similarity**
2. **Manual calculation would be redundant and slow**
3. **The function was implementing something already handled**

We now use cosine similarity for:
- **Validation**: Ensuring Databricks scores are correct
- **Debugging**: Troubleshooting similarity issues
- **Quality Assurance**: Verifying embedding quality

This approach gives us the best of both worlds: fast, scalable similarity search from Databricks, plus validation capabilities for debugging and quality assurance. 