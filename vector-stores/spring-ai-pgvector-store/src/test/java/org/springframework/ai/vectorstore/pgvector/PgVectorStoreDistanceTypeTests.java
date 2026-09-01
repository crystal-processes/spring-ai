/*
 * Copyright 2023-present the original author or authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.springframework.ai.vectorstore.pgvector;

import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.SQLException;
import java.util.List;

import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;

import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.jdbc.core.PreparedStatementCreator;
import org.springframework.jdbc.core.ResultSetExtractor;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * Unit tests for {@link PgVectorStore} configurable distance type behavior.
 *
 */
class PgVectorStoreDistanceTypeTests {

	private static final TestPgDistanceType CUSTOM_DISTANCE_TYPE = new TestPgDistanceType("CUSTOM", "<custom>",
			"custom_ops",
			"SELECT *, embedding CUSTOM ? AS distance FROM %s WHERE embedding CUSTOM ? < ? %s ORDER BY distance LIMIT ? ");

	@Test
	void shouldUseCosineDistanceByDefault() {
		// Given
		var jdbcTemplate = mock(JdbcTemplate.class);
		var embeddingModel = mock(EmbeddingModel.class);

		// When
		var vectorStore = PgVectorStore.builder(jdbcTemplate, embeddingModel).build();

		// Then
		assertThat(vectorStore.getDistanceType()).isEqualTo(PgVectorStore.COSINE_DISTANCE);
		assertThat(vectorStore.getDistanceType().operator()).isEqualTo("<=>");
		assertThat(vectorStore.getDistanceType().index()).isEqualTo("vector_cosine_ops");
	}

	@Test
	void shouldUseCustomDistanceType() {
		// Given
		var jdbcTemplate = mock(JdbcTemplate.class);
		var embeddingModel = mock(EmbeddingModel.class);

		// When
		var vectorStore = PgVectorStore.builder(jdbcTemplate, embeddingModel)
			.distanceType(CUSTOM_DISTANCE_TYPE)
			.build();

		// Then
		assertThat(vectorStore.getDistanceType()).isEqualTo(CUSTOM_DISTANCE_TYPE);
		assertThat(vectorStore.getDistanceType().operator()).isEqualTo("<custom>");
		assertThat(vectorStore.getDistanceType().index()).isEqualTo("custom_ops");
	}

	@Test
	void similaritySearchShouldUseConfiguredDistanceTypeOperator() throws SQLException {
		// Given
		var jdbcTemplate = mock(JdbcTemplate.class);
		var embeddingModel = mock(EmbeddingModel.class);
		when(embeddingModel.dimensions()).thenReturn(3);
		when(embeddingModel.embed(anyString())).thenReturn(new float[] { 0.1f, 0.2f, 0.3f });
		when(jdbcTemplate.query(any(PreparedStatementCreator.class), any(ResultSetExtractor.class)))
			.thenReturn(List.of());

		var vectorStore = PgVectorStore.builder(jdbcTemplate, embeddingModel)
			.distanceType(PgVectorStore.COSINE_DISTANCE)
			.initializeSchema(false)
			.build();

		var request = SearchRequest.builder().query("test query").topK(5).similarityThresholdAll().build();

		// When
		vectorStore.doSimilaritySearch(request);

		// Then
		var sqlCaptor = ArgumentCaptor.forClass(PreparedStatementCreator.class);
		verify(jdbcTemplate).query(sqlCaptor.capture(), any(ResultSetExtractor.class));
		Connection connection = mock(Connection.class);
		when(connection.prepareStatement(
				"SELECT *, embedding <=> ? AS distance FROM public.vector_store WHERE embedding <=> ? < ?  ORDER BY distance LIMIT ? "))
			.thenReturn(mock(PreparedStatement.class));
		PreparedStatementCreator sql = sqlCaptor.getValue();
		sql.createPreparedStatement(connection);

		// Verify that the default cosine distance operator is used in the SQL
		verify(connection, times(1)).prepareStatement(
				"SELECT *, embedding <=> ? AS distance FROM public.vector_store WHERE embedding <=> ? < ?  ORDER BY distance LIMIT ? ");
	}

	/**
	 * Test implementation of {@link PgDistanceType}.
	 */
	private record TestPgDistanceType(String name, String operator, String index,
			String similaritySearchSqlTemplate) implements PgDistanceType {

	}

}
