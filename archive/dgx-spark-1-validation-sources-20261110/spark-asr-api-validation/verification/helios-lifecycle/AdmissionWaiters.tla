------------------------- MODULE AdmissionWaiters -------------------------
EXTENDS Naturals, Sequences, FiniteSets

CONSTANT HostCapacity
VARIABLE state

Empty == [location |-> 0, reader |-> 0, cancelled |-> 0]
Posted == [location |-> 1, reader |-> 1, cancelled |-> 0]
Owned(row) == row.location > 0 \/ row.reader = 1
Bit(value) == IF value THEN 1 ELSE 0
HostOwners(st) == Cardinality({r \in 1..2 : Owned(st.rows[r])})
MessageOwners(st) == Cardinality({r \in 1..2 : st.rows[r].location = 1})
Initial == [capacity |-> HostCapacity, host |-> HostCapacity, message |-> 1,
            closed |-> 0, waiting |-> 0, assignedHost |-> 0, assignedMessage |-> 0,
            rows |-> [r \in 1..2 |-> Empty]]
Init == state = Initial

\* A released semaphore credit may already belong to an unpolled acquire.
Wake(st) ==
    IF st.closed = 1 THEN st
    ELSE IF st.waiting = 1 /\ st.assignedHost = 0 /\ st.host > 0
         THEN [st EXCEPT !.host = @ - 1, !.assignedHost = 1]
    ELSE IF st.waiting = 2 /\ st.assignedMessage = 0 /\ st.message > 0
         THEN [st EXCEPT !.message = @ - 1, !.assignedMessage = 1]
    ELSE st

DiscardWaiter(before, after) ==
    /\ before.waiting \in {1, 2}
    /\ after = [before EXCEPT
         !.host = @ + Bit(before.waiting = 2) + before.assignedHost,
         !.message = @ + before.assignedMessage,
         !.waiting = 0, !.assignedHost = 0, !.assignedMessage = 0]

PollWaiter(before, after) ==
    /\ before.waiting \in {1, 2}
    /\ IF before.closed = 1 THEN DiscardWaiter(before, after)
       ELSE IF before.waiting = 1 /\ before.host = 0 /\ before.assignedHost = 0
            THEN after = before
       ELSE LET held == [before EXCEPT !.waiting = 2, !.assignedHost = 0,
                    !.host = IF before.waiting = 1 /\ before.assignedHost = 0 THEN @ - 1 ELSE @]
            IN IF held.message = 0 /\ held.assignedMessage = 0 THEN after = held
               ELSE after = [held EXCEPT !.waiting = 0, !.assignedMessage = 0,
                    !.message = IF held.assignedMessage = 0 THEN @ - 1 ELSE @,
                    !.rows[2] = Posted]

BeginWaiter(before, after) ==
    /\ before.closed = 0 /\ before.waiting = 0 /\ ~Owned(before.rows[2])
    /\ PollWaiter([before EXCEPT !.waiting = 1], after)

AdmitBlocker(before, after) ==
    /\ before.closed = 0 /\ ~Owned(before.rows[1])
    /\ before.host > 0 /\ before.message > 0
    /\ after = [before EXCEPT !.host = @ - 1, !.message = @ - 1, !.rows[1] = Posted]

DropReader(before, after, r) ==
    /\ before.rows[r].reader = 1
    /\ LET old == before.rows[r]
           row == IF old.location = 0 THEN Empty
                  ELSE [old EXCEPT !.reader = 0, !.cancelled = 1]
           released == [before EXCEPT !.rows[r] = row,
                         !.host = @ + Bit(~Owned(row))]
       IN after = Wake(released)

DropProducer(before, after, r) ==
    /\ before.rows[r].location > 0
    /\ LET old == before.rows[r]
           row == IF old.reader = 0 THEN Empty ELSE [old EXCEPT !.location = 0]
           released == [before EXCEPT !.rows[r] = row,
                         !.host = @ + Bit(~Owned(row)), !.message = @ + Bit(old.location = 1)]
       IN after = Wake(released)

Enqueue(before, after, r) ==
    /\ before.rows[r].location = 1
    /\ after = Wake([before EXCEPT !.rows[r].location = 2, !.message = @ + 1])

Close(before, after) == before.closed = 0 /\ after = [before EXCEPT !.closed = 1]
Reject(before, after) ==
    /\ before.closed = 1 \/ before.host = 0 \/ before.message = 0
    /\ after = before

Next == AdmitBlocker(state, state') \/ BeginWaiter(state, state')
     \/ PollWaiter(state, state') \/ DiscardWaiter(state, state') \/ Close(state, state')
     \/ Reject(state, state')
     \/ (\E r \in 1..2 : DropReader(state, state', r) \/ DropProducer(state, state', r)
                            \/ Enqueue(state, state', r))

WellTyped(st) ==
    /\ st.capacity \in {1, 2} /\ st.host \in 0..st.capacity /\ st.message \in 0..1
    /\ st.closed \in 0..1 /\ st.waiting \in 0..2
    /\ st.assignedHost \in 0..1 /\ st.assignedMessage \in 0..1
    /\ st.rows \in [1..2 -> [location : 0..2, reader : 0..1, cancelled : 0..1]]

Credits(st) ==
    /\ st.host + HostOwners(st) + Bit(st.waiting = 2) + st.assignedHost = st.capacity
    /\ st.message + MessageOwners(st) + st.assignedMessage = 1
    /\ st.assignedHost <= Bit(st.waiting = 1)
    /\ st.assignedMessage <= Bit(st.waiting = 2)
    /\ Owned(st.rows[2]) => st.waiting = 0

Cancellation(st) ==
    \A r \in 1..2 : st.rows[r].cancelled = Bit(st.rows[r].location > 0 /\ st.rows[r].reader = 0)

TypeOK == WellTyped(state)
CreditsRetained == Credits(state)
DroppedReadersCancel == Cancellation(state)

Projection(st) == <<st.capacity, st.host, st.message, st.closed, st.waiting,
                    st.assignedHost, st.assignedMessage,
                    st.rows[1].location, st.rows[1].reader, st.rows[1].cancelled,
                    st.rows[2].location, st.rows[2].reader, st.rows[2].cancelled>>
Decode(v) == [capacity |-> v[1], host |-> v[2], message |-> v[3], closed |-> v[4],
    waiting |-> v[5], assignedHost |-> v[6], assignedMessage |-> v[7],
    rows |-> [r \in 1..2 |-> [location |-> v[8 + 3 * (r - 1)],
        reader |-> v[9 + 3 * (r - 1)], cancelled |-> v[10 + 3 * (r - 1)]]]]

AllowedEdge(edge) ==
    LET before == Decode(edge[1])
        action == edge[2]
        after == Decode(edge[3])
    IN /\ Projection(before) = edge[1] /\ Projection(after) = edge[3]
       /\ WellTyped(before) /\ WellTyped(after) /\ Credits(before) /\ Credits(after)
       /\ Cancellation(before) /\ Cancellation(after)
       /\ CASE action[1] = 1 -> AdmitBlocker(before, after)
         [] action[1] = 2 -> BeginWaiter(before, after)
         [] action[1] = 3 -> PollWaiter(before, after)
         [] action[1] = 4 -> DiscardWaiter(before, after)
         [] action[1] = 5 -> DropReader(before, after, action[2] + 1)
         [] action[1] = 6 -> DropProducer(before, after, action[2] + 1)
         [] action[1] = 7 -> Enqueue(before, after, action[2] + 1)
         [] action[1] = 8 -> Close(before, after)
         [] action[1] = 9 -> Reject(before, after)
         [] OTHER -> FALSE

Spec == Init /\ [][Next]_state
=============================================================================
