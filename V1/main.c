#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define STATUS_TYPES 5

typedef struct {
    double response_time;
    int status_code;
    double data_transferred;
} LogEntry;

int status_codes[STATUS_TYPES] = {200, 301, 400, 404, 500};

void generate_logs(LogEntry *logs, long N) {
    for(long i = 0; i < N; i++) {
        logs[i].response_time = rand() % 1000 + (rand() % 100) / 100.0;
        logs[i].status_code = status_codes[rand() % STATUS_TYPES];
        logs[i].data_transferred = rand() % 5000;
    }
}

int main() {

    long N;
    int threads;

    N = 10000000;




    threads = 5;

    //printf("Enter number of log entries: ");
    //scanf("%ld", &N);

    //printf("Enter number of threads: ");
    //scanf("%d", &threads);

    omp_set_num_threads(threads);

    LogEntry *logs = malloc(sizeof(LogEntry) * N);
    double *prefix_data = malloc(sizeof(double) * N);

    generate_logs(logs, N);

    double start, end;

    // 
    // SERIAL EXECUTION
    // 
    start = omp_get_wtime();

    double total_data = 0;
    double total_response = 0;
    double max_response = 0;
    int status_count[STATUS_TYPES] = {0};

    for(long i = 0; i < N; i++) {
        total_data += logs[i].data_transferred;
        total_response += logs[i].response_time;

        if(logs[i].response_time > max_response)
            max_response = logs[i].response_time;

        for(int j = 0; j < STATUS_TYPES; j++)
            if(logs[i].status_code == status_codes[j])
                status_count[j]++;
    }

    double avg = total_response / N;

    double resp_time_variance = 0;
    for(long i = 0; i < N; i++)
        resp_time_variance += pow(logs[i].response_time - avg, 2);

    resp_time_variance /= N;
    double stddev = sqrt(resp_time_variance);

    end = omp_get_wtime();
    double serial_time = end - start;

    // 
    // PARALLEL EXECUTION
    // 
    start = omp_get_wtime();

    double p_total_data = 0;
    double p_total_response = 0;
    double p_max_response = 0;
    int p_status_count[STATUS_TYPES] = {0};

    #pragma omp parallel
    {
        int local_status[STATUS_TYPES] = {0};

        #pragma omp for reduction(+:p_total_data,p_total_response) reduction(max:p_max_response) schedule(dynamic)
        for(long i = 0; i < N; i++) {
            p_total_data += logs[i].data_transferred;
            p_total_response += logs[i].response_time;

            if(logs[i].response_time > p_max_response)
                p_max_response = logs[i].response_time;

            for(int j = 0; j < STATUS_TYPES; j++)
                if(logs[i].status_code == status_codes[j])
                    local_status[j]++;
        }

        #pragma omp critical
        {
            for(int j = 0; j < STATUS_TYPES; j++)
                p_status_count[j] += local_status[j];
        }
    }

    double p_avg = p_total_response / N;

    double p_resp_time_variance = 0;

    #pragma omp parallel for reduction(+:p_resp_time_variance)
    for(long i = 0; i < N; i++)
        p_resp_time_variance += pow(logs[i].response_time - p_avg, 2);

    p_resp_time_variance /= N;
    double p_stddev = sqrt(p_resp_time_variance);

    end = omp_get_wtime();
    double parallel_time = end - start;

    // 
    // PREFIX SUM USING TASKS
    // 
    #pragma omp parallel
    {
        #pragma omp single
        {
            prefix_data[0] = logs[0].data_transferred;

            for(long i = 1; i < N; i++) {
                #pragma omp task firstprivate(i)
                prefix_data[i] = prefix_data[i-1] + logs[i].data_transferred;
            }
        }
    }

    // Ensure all tasks are complete
    #pragma omp taskwait

// ==============================
// PREFIX SUM USING TASKS 
// ==============================
#pragma omp parallel
{
    #pragma omp single
    {
        prefix_data[0] = logs[0].data_transferred;

        for(long i = 1; i < N; i++) {
            #pragma omp task firstprivate(i)
            prefix_data[i] = prefix_data[i-1] + logs[i].data_transferred;
        }
    }
}

// ==============================
// PREFIX-BASED ANALYTICS (PARALLELIZED)
// ==============================

// 1️⃣ Verify total data
printf("\nTotal Data via Reduction : %f\n", p_total_data);
printf("Total Data via Prefix    : %f\n", prefix_data[N-1]);

// ==============================
// 2️⃣ PARALLEL THRESHOLD DETECTION
// ==============================
double threshold_percent = 0.75;
double threshold = threshold_percent * prefix_data[N-1];

long threshold_index = N;

#pragma omp parallel
{
    long local_min = N;

    #pragma omp for nowait
    for(long i = 0; i < N; i++) {
        if(prefix_data[i] >= threshold && i < local_min)
            local_min = i;
    }

    #pragma omp critical
    {
        if(local_min < threshold_index)
            threshold_index = local_min;
    }
}

if(threshold_index == N)
    threshold_index = -1;

printf("\n%.0f%% of total traffic crossed at index: %ld\n",
       threshold_percent * 100, threshold_index);


// ==============================
// 3️⃣ PARALLEL PEAK GROWTH DETECTION
// ==============================
double max_growth = 0;
long growth_index = -1;

#pragma omp parallel
{
    double local_max = 0;
    long local_index = -1;

    #pragma omp for nowait
    for(long i = 1; i < N; i++) {
        double growth = prefix_data[i] - prefix_data[i-1];

        if(growth > local_max) {
            local_max = growth;
            local_index = i;
        }
    }

    #pragma omp critical
    {
        if(local_max > max_growth) {
            max_growth = local_max;
            growth_index = local_index;
        }
    }
}

    printf("Peak growth of %f occurred at index %ld\n",
           max_growth, growth_index);

    long query_index;
    query_index = N/2;
    printf("\nEnter index to query cumulative bandwidth: ");
    //scanf("%ld", &query_index);

    if(query_index >= 0 && query_index < N)
        printf("Cumulative bandwidth at index %ld = %f\n",
               query_index, prefix_data[query_index]);
    else
        printf("Invalid index.\n");

    // ==============================
    // FINAL OUTPUT
    // ==============================
    printf("\n===== PERFORMANCE =====\n");
    printf("Serial Time   : %f seconds\n", serial_time);
    printf("Parallel Time : %f seconds\n", parallel_time);
    printf("Speedup       : %f\n", serial_time / parallel_time);

    printf("\n===== STATISTICS =====\n");
    printf("Average Response : %f\n", p_avg);
    printf("Max Response     : %f\n", p_max_response);
    printf("Std Deviation    : %f\n", p_stddev);

    free(logs);
    free(prefix_data);

    return 0;
}