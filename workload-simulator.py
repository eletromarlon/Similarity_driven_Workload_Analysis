import psycopg2
import time
import random
import csv
import threading
from datetime import datetime

# ==============================
# CONFIGURAÇÕES DE CONEXÃO
# ==============================
DB_CONFIG = {
    "dbname": "postgres",
    "user": "postgres",
    "password": "Abc.123*",
    "host": "localhost",
    "port": 5432
}

LOG_FILE = "query_log.csv"
LOG_LOCK = threading.Lock()  # para evitar escrita simultânea no arquivo

# ==============================
# CONSULTAS (ampliadas + similares)
# ==============================

SELECT_QUERIES = [
    # Simples
    "SELECT first_name, last_name FROM customer WHERE customer_id = %s;",
    "select first_name, last_name from customer where customer_id=%s;",  # similar

    # relatório
    "SELECT title, release_year FROM film ORDER BY release_year DESC LIMIT 10;",
    "select title, release_year from film order by release_year desc limit 10;",

    # join simples
    "SELECT c.first_name, c.last_name, a.address FROM customer c JOIN address a ON c.address_id = a.address_id LIMIT 5;",

    # join com filtro
    "SELECT f.title, c.name AS category FROM film f "
    "JOIN film_category fc ON f.film_id = fc.film_id "
    "JOIN category c ON fc.category_id = c.category_id "
    "WHERE c.name = 'Action' LIMIT 10;",

    # agregação
    "SELECT s.first_name, s.last_name, COUNT(r.rental_id) AS total_rentals "
    "FROM staff s JOIN rental r ON s.staff_id = r.staff_id "
    "GROUP BY s.staff_id;"
]

INSERT_QUERIES = [
    "INSERT INTO rental (rental_date, inventory_id, customer_id, staff_id) VALUES (NOW(), %s, %s, %s);",
    "insert into rental (rental_date, inventory_id, customer_id, staff_id) values (now(), %s, %s, %s);",

    "INSERT INTO payment (customer_id, staff_id, rental_id, amount, payment_date) "
    "VALUES (%s, %s, %s, %s, NOW());",

    "INSERT INTO customer (store_id, first_name, last_name, email, address_id, activebool, create_date, last_update) "
    "VALUES (%s, %s, %s, %s, %s, TRUE, NOW(), NOW());",

    "INSERT INTO film_category (film_id, category_id, last_update) VALUES (%s, %s, NOW());"
]

UPDATE_QUERIES = [
    "UPDATE customer SET last_update = NOW() WHERE customer_id = %s;",
    "update customer set last_update=now() where customer_id=%s;",

    "UPDATE film SET rental_rate = rental_rate + 0.1 WHERE film_id = %s;",

    "UPDATE inventory SET last_update = NOW() WHERE inventory_id = %s;",

    "UPDATE staff SET active = NOT active WHERE staff_id = %s;"
]

DELETE_QUERIES = [
    "DELETE FROM payment WHERE payment_id = %s;",
    "delete from payment where payment_id=%s;",

    "DELETE FROM rental WHERE rental_id = %s;",

    "DELETE FROM film_category WHERE film_id = %s AND category_id = %s;"
]


# ==============================
# FUNÇÃO DE LOG
# ==============================
def log_query(timestamp, qtype, query, duration):
    with LOG_LOCK:
        with open(LOG_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, qtype, query, round(duration, 5)])


# ==============================
# EXECUTAR QUERY
# ==============================
def run_query(cur, conn, qtype, query):
    params = None

    # gerar parâmetros
    n_params = query.count("%s")
    if n_params > 0:
        params = tuple(random.randint(1, 10) for _ in range(n_params))

    # medir duração
    start = time.time()
    try:
        cur.execute(query, params)
        if qtype != "SELECT":
            conn.commit()
    except Exception as e:
        conn.rollback()
        print(f"[ERRO] {e} -> {query}")
        return None

    duration = time.time() - start
    return duration


# ==============================
# TRABALHADOR (THREAD)
# ==============================
def worker_thread(thread_id, runtime):
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    start_time = time.time()

    print(f"Thread {thread_id} iniciada.")

    while time.time() - start_time < runtime:
        qtype = random.choice(["SELECT", "INSERT", "UPDATE", "DELETE"])

        if qtype == "SELECT":
            query = random.choice(SELECT_QUERIES)
        elif qtype == "INSERT":
            query = random.choice(INSERT_QUERIES)
        elif qtype == "UPDATE":
            query = random.choice(UPDATE_QUERIES)
        else:
            query = random.choice(DELETE_QUERIES)

        duration = run_query(cur, conn, qtype, query)
        if duration is not None:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_query(timestamp, qtype, query, duration)

        # pequena pausa entre queries
        time.sleep(random.uniform(0.1, 0.5))

    cur.close()
    conn.close()
    print(f"Thread {thread_id} finalizada.")


# ==============================
# FUNÇÃO PRINCIPAL
# ==============================
def simulate_load(users=10, runtime=60):
    # criar/limpar CSV
    with open(LOG_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "query_type", "query_text", "duration"])

    threads = []

    for i in range(users):
        t = threading.Thread(target=worker_thread, args=(i, runtime))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    print("✔ Simulação concluída.")


# ==============================
# EXECUTAR
# ==============================
if __name__ == "__main__":
    simulate_load(users=20, runtime=120)
