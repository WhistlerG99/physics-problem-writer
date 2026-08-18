NOTES_DIR=notes/superluminal-bh/
OUTPUT_DIR=results/superluminal-bh/
TRIAL_NUM=1

WRITER_MODEL=gpt-4o
STUDENT_MODEL=gpt-4o


python physics_writer_agent.py \
    $NOTES_DIR \
    --students 5 \
    --max-correct 1 \
    --rounds 5 \
    --output $OUTPUT_DIR/trial_$TRIAL_NUM \
    --summary-dir $OUTPUT_DIR/notes-summary \
    --reviewer-samples 2 \
    --writer-model $WRITER_MODEL \
    --student-model $STUDENT_MODEL
